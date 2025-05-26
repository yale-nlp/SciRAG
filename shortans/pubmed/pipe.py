from llms import LanguageModel, get_model
import argparse
import json
import os
import sys
from typing import Optional
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from concurrent.futures import ThreadPoolExecutor, as_completed
import instructions as instructions
import re
from web_search import search_snippet_via_query_w_citation_expansion
from FlagEmbedding import FlagReranker
import json
class TreeNode:
    def __init__(self, query, answer, state, parent=None):
        self.node = {
            'query': query,
            'answer': answer,
            'state': state
        }
        self.children = []
        self.parent = parent  
        self.references = []
        
    def set_references(self, refs):
        self.references = refs
        
    def add_reference(self, ref_item):
        """
        ref_item: dict, like:
         { "paperId": xxx, "title": "...", "text": "...", ........}
        """
        self.references.append(ref_item)

    def merge_references(self, new_refs):
        """
        add new_refs (list[dict]) to self.references
        """
        existing_ids = { ref["paperId"] for ref in self.references if "paperId" in ref }
        for new_r in new_refs:
            pid = new_r.get("paperId", None)
            if pid and pid in existing_ids:
                for idx, old_ref in enumerate(self.references):
                    if old_ref.get("paperId") == pid:
                        self.references[idx] = new_r 
            else:
                self.references.append(new_r)
                existing_ids.add(pid)
                
    def set_query(self, query):
        self.node['query'] = query


    def set_answer(self, answer):
        self.node['answer'] = answer

    def set_state(self, state):
        if isinstance(state, bool):
            self.node['state'] = state
        else:
            raise ValueError("state must be a boolean value.")

    def add_child(self, child_node):
        if isinstance(child_node, TreeNode):
            child_node.parent = self  
            self.children.append(child_node)
        else:
            raise ValueError("child_node must be an instance of TreeNode.")

    def get_node(self):
        return self.node

    def get_children(self):
        return self.children

    def get_parent(self):
        return self.parent  

    def get_root(self):

        if self.parent is None:
            return self  
        return self.parent.get_root()  

    def __repr__(self):
        return f"TreeNode(query={self.node['query']}, " \
               f"answer={self.node['answer']}, state={self.node['state']})"

class pipeline:
    def __init__(self,  top_n=8) -> None:
        self.top_n = top_n
    
    def gen_outline(self,item, model,max_tokens=500):
        client=get_model(model)
        print(item["input"])
        input_query = instructions.Outline_Generation_Instruction.format_map({"question": item["input"]})
        outputs = client.chat(messages=input_query, max_tokens=max_tokens)
        raw_output = [t.split("[Response_End]")[0] for t in outputs.split("[Response_Start]") if "[Response_End]" in t][0] if "[Response_End]" in outputs else outputs
        cost = calculate_openai_api_cost(len(input_query.split(" ")),len(outputs.split(" ")), model)
        return raw_output,cost
    
    def initial_result(self,item, model,outline, max_tokens=3000):
        client=get_model(model)
        ctxs = ""
        for doc_idx, doc in enumerate(item["ctxs"][:self.top_n]):
            if "title" in doc and len(doc["title"]) > 0:
                ctxs += "[{0}] Title: {1} Text: {2}\n".format(doc_idx, doc["title"], doc["text"])
            else:
                ctxs += "[{0}] {1}\n".format(doc_idx,  doc["text"])
        input_query = instructions.initial_generation_w_references_zero_shot.format_map({"context": ctxs, "input": item["input"],"outline": outline})
        outputs = client.chat(messages=input_query, max_tokens=max_tokens)
        raw_output = [t.split("[Response_End]")[0] for t in outputs.split("[Response_Start]") if "[Response_End]" in t][0] if "[Response_End]" in outputs else outputs
        cost = calculate_openai_api_cost(len(input_query.split(" ")),len(outputs.split(" ")), model)
        return raw_output,cost
    
    def additional_result(self,path,query, model,context, max_tokens):
        client=get_model(model)
        ctxs = context
        input_query = instructions.additional_generation_w_references_zero_shot.format_map({"path":path, "query": query,"context": ctxs})
        outputs = client.chat(messages=input_query, max_tokens=max_tokens)
        cost=calculate_openai_api_cost(len(input_query.split(" ")),len(outputs.split(" ")), model)
        raw_output = [t.split("[Response_End]")[0] for t in outputs.split("[Response_Start]") if "[Response_End]" in t][0] if "[Response_End]" in outputs else outputs
        return raw_output,cost
        

    def symbolic_reason_rerank(self, papers, path, query, model, max_tokens=5000):
        strs = 'Papers:\n'
        for idx, paper in enumerate(papers, start=1):
            strs += f"[{idx}] \n {paper['text']}\n\n"
        t_cost=0
        client = get_model(model)  
        sys_msg = instructions.system_reasoning.format_map({"path": path, "query": query})

        # Step 1
        step1_prompt = instructions.step1_reason_prompt.format_map({"paper_text": strs, "query_text": query})
        step1_outputs = client.chat(messages=step1_prompt, system_msg=sys_msg, temperature=0.0, max_tokens=max_tokens)
        cost=calculate_openai_api_cost(len(step1_prompt.split(" ")),len(step1_outputs.split(" ")), model)
        t_cost+=cost
        # Step 2
        step2_prompt = instructions.step2_reason_prompt.format_map({"step1_result_json": step1_outputs, "query": query})
        step2_outputs = client.chat(messages=step2_prompt, system_msg=sys_msg, temperature=0.0, max_tokens=max_tokens)
        cost=calculate_openai_api_cost(len(step2_prompt.split(" ")),len(step2_outputs.split(" ")), model)
        t_cost+=cost
        # Step 3
        step3_prompt = instructions.step3_reason_prompt.format_map({
            "step2_relationships_json": step2_outputs,
            "step1_result_json": step1_outputs,
            "query": query,
            "path": path
        })
        step3_outputs = client.chat(
            messages=step3_prompt,
            system_msg=sys_msg,
            temperature=0.0,
            max_tokens=max_tokens,
            json_mode=True  
        )
        cost=calculate_openai_api_cost(len(step3_prompt.split(" ")),len(step3_outputs.split(" ")), model)
        t_cost+=cost
        try:
            step3_result = json.loads(step3_outputs)
        except json.JSONDecodeError:
            print("Step 3: JSON parse error. Raw response:\n", step3_outputs)
            step3_result = step3_outputs

        final_selection = step3_result.get("final_selection", [])
        final_selection = sorted(final_selection, key=lambda x: x.get("rank", 999999))

        reranked_papers = []
        for item in final_selection:
            paper_idx = item.get("paper_index", -1) - 1
            reasons=item.get("justification",'')
            if 0 <= paper_idx < len(papers):
                papers[paper_idx]["rationale"]=reasons
                reranked_papers.append(papers[paper_idx])

        return reranked_papers, t_cost

    def additional_references_expansion(self,path, query, model, max_tokens,topn=10):
        paper_list = []
        total_cost=0
        all_papers_data,status,cost_d = search_snippet_via_query_w_citation_expansion(query=query,max_paper_num=5,min_citation_count=10)
        total_cost+=cost_d
        ctxs = ""
        if all_papers_data is None:
            print(query)
            reranked_papers = []
        else:
            seen_paper_ids = set()  
            paper_list_filtered = []  

            for paper_data in all_papers_data:
                paper = paper_data['paper']
                abstract = paper.get("abstract")
                snippet_text = paper.get("snippet_text")

                if abstract is not None and snippet_text is not None:
                    paper["text"] = f"Abstract:\n{abstract}\n Related text:\n{snippet_text}"
                elif abstract is not None:
                    paper["text"] = f"Abstract:\n{abstract}"
                elif snippet_text is not None:
                    paper["text"] = f"Related text:\n{snippet_text}"
                else:
                    paper["text"] = ""
                    
                if paper["paperId"] not in seen_paper_ids:
                    seen_paper_ids.add(paper["paperId"])
                    paper_list_filtered.append(paper)

                if "references" in paper_data and paper_data["references"] is not None:
                    for ref in paper_data["references"]:
                        ref_data = ref.copy()
                        ref_data["text"] = ref.get("abstract", "")

                        if ref_data["paperId"] not in seen_paper_ids:
                            seen_paper_ids.add(ref_data["paperId"])
                            paper_list_filtered.append(ref_data)

                if "citations" in paper_data and paper_data["citations"] is not None:
                    for citation in paper_data["citations"]:
                        citation_data = citation.copy()
                        citation_data["text"] = citation.get("abstract", "")

                        if citation_data["paperId"] not in seen_paper_ids:
                            seen_paper_ids.add(citation_data["paperId"])
                            paper_list_filtered.append(citation_data)

            paper_list = paper_list_filtered  

            reranker=FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
            reranked_papers = []
            for paper in paper_list:
                paper_text = f"{paper['title']}\n{paper['text']}"
                score = reranker.compute_score([[query, paper_text]], batch_size=4)[0]
                paper['score'] = score  
                reranked_papers.append(paper)

            top_papers = sorted(reranked_papers, key=lambda x: x['score'], reverse=True)[:min(topn,len(reranked_papers))]
            if status:
                reranked_papers,cost = self.symbolic_reason_rerank(top_papers, path, query, model)
            else:
                reranked_papers = top_papers
                cost=0
            total_cost+=cost
            for idx, paper in enumerate(reranked_papers, start=1):
                ctxs += f"[{idx}] \n {paper['text']}\n The rationale for choosing this paper:{paper['rationale']}\n\n"
        

        outputs,cost = self.additional_result(path,query, model, ctxs, max_tokens)
        total_cost+=cost
        return outputs,  reranked_papers,total_cost
    
    def initial_critic(self,answer,query,guidance, model,max_tokens=500):
        client=get_model(model)
        query_pattern = r"\(\d+\)\s*(.+)"  
        input_query = instructions.Gap_Critic_Prompt_initial.format_map({"guidance": guidance,"answer": answer, "query" : query})
        outputs = client.chat(messages=input_query, max_tokens=max_tokens)
        cost = calculate_openai_api_cost(len(input_query.split(" ")),len(outputs.split(" ")), model)
        if "[end]terminate" in outputs:
            return True,None, cost
        else:
            matches = re.findall(query_pattern, outputs)
            queries = [match.strip() for match in matches]
            return False,queries ,cost   
        
    def get_query_path(self,father):
        queries = []
        current = father
        while current is not None:
            queries.append(current.node['query'])
            current = current.get_parent()
        queries.reverse()  
        return "->".join(queries)    
    
    def gap_critic(self,father,model,depth,max_depth, max_tokens=500):
        depth+=1
        local_cost = 0
        path=self.get_query_path(father)
        if depth>=max_depth:
            ans,refs,cost=self.additional_references_expansion(path, father.node["query"], model,max_tokens=1000)
            father.set_answer(ans)
            father.set_state(True)
            father.set_references(refs)
            local_cost += cost
            return father,local_cost
        else:
            ans,refs,cost=self.additional_references_expansion(path, father.node["query"], model,max_tokens=1000)
            father.set_answer(ans)
            father.set_state(True)
            father.set_references(refs)
            local_cost += cost
            client=get_model(model)
            query_pattern = r"\(\d+\)\s*(.+)"  
            input_query = instructions.Gap_Critic_Prompt_later.format_map({"path":path, "query": father.node["query"] ,"answer": ans})
            outputs = client.chat(messages=input_query, max_tokens=max_tokens)
            new_cost = calculate_openai_api_cost(len(input_query.split(" ")),len(outputs.split(" ")), model)
            local_cost += new_cost
            if "[end]terminate" in outputs:
                return father, local_cost
            else:
                matches = re.findall(query_pattern, outputs)
                queries = [match.strip() for match in matches]
                for query in queries:
                    child_node = TreeNode(query=query, answer='', state=False)
                    child_node, child_cost = self.gap_critic(child_node, model, depth, max_depth, max_tokens)
                    local_cost += child_cost
                    child_node.set_state(True)
                    father.add_child(child_node)
                return father, local_cost


    def bottom_up_aggregate(self,guideline,node: TreeNode, model, maxtokens=1000):

        client = get_model(model)
        local_cost = 0 
        if not node.children:
            return node.node['answer'], node.references,local_cost

        child_info_list = []
        for child in node.children:
            child_answer, child_refs, child_cost = self.bottom_up_aggregate(guideline,child, model, maxtokens)
            local_cost += child_cost
            child.set_answer(child_answer)
            child.set_references(child_refs)
 
            child_info_list.append({
                "query": child.node["query"],
                "answer": child_answer,
                "references": child_refs
            })

        child_answers_str = ""
        for idx, cinfo in enumerate(child_info_list, start=1):
            child_answers_str += f"Child Query {idx}:\n{cinfo['query']}\nAnswer:\n{cinfo['answer']}\n\n"
            
        for cinfo in child_info_list:
            node.merge_references(cinfo["references"])  
        ctxs = ""  
        for idx, paper in enumerate(node.references, start=1):
            text = paper.get("text", "").strip()  
            if text:  
                ctxs += f"[{idx}]\n{text}\n\n"
        if node.parent==None:
            prompt_str = instructions.final_prompt_zero_shot.format_map({
                "query": node.node["query"],           
                "answer": node.node["answer"],         
                "supplement": child_answers_str, 
                "context":ctxs,
                "outline":guideline
            })
        else:
            path=self.get_query_path(node)
            prompt_str = instructions.branch_prompt_zero_shot.format_map({
                "query": node.node["query"],           
                "answer": node.node["answer"],         
                "supplement": child_answers_str, 
                "context":ctxs,
                "path":path
            })

        merged_answer = client.chat(messages=prompt_str, max_tokens=maxtokens)
        cost_call = calculate_openai_api_cost(len(prompt_str.split(" ")),len(merged_answer.split(" ")),model)
        local_cost += cost_call
        raw_answer = self._clean_llm_output(merged_answer)  

        node.node["answer"] = raw_answer 
        return node.node['answer'], node.references, local_cost

    def _clean_llm_output(self,llm_response: str) -> str:
        if "[Response_End]" in llm_response:
            segments = llm_response.split("[Response_Start]")
            extracted = []
            for seg in segments:
                if "[Response_End]" in seg:
                    extracted.append(seg.split("[Response_End]")[0])
            if extracted:
                return extracted[0].strip()
            else:
                return llm_response.strip()
        else:
            return llm_response.strip()

    def _get_additional_answers(self, node):
        additional_answer = ""
        
        for child in node.get_children():
            additional_answer += f"Query:\n{child.node['query']}\n Answer:\n{child.node['answer']}\n\n"
            additional_answer += self._get_additional_answers(child)
        
        return additional_answer

    def reflective_refine(self,root,ans,refs,model,guideline,max_tokens=5000): 
        client=get_model(model)
        cost=0
        ctxs = ""
        for idx, paper in enumerate(refs, start=1):
            text = paper.get("text", "").strip()  
            if text:  
                ctxs += f"[{idx}]\n{text}\n\n"
        input_query = instructions.ref_feedback.format_map({"question": root.node["query"], "answer": ans, "outline": guideline})
        outputs = client.chat(messages=input_query, max_tokens=1000)
        cost+=calculate_openai_api_cost(len(input_query.split(" ")),len(outputs.split(" ")), model)
        raw_output = [t.split("[Response_End]")[0] for t in outputs.split("[Response_Start]") if "[Response_End]" in t][0] if "[Response_End]" in outputs else outputs
        if "[terminate]" in raw_output:
            return ans
        input_query = instructions.edit_feedback.format_map({"question": root.node["query"], "original_answer": ans, "feedback":raw_output,"outline": guideline, "references": ctxs})
        outputs = client.chat(messages=input_query, max_tokens=max_tokens)
        cost+=calculate_openai_api_cost(len(input_query.split(" ")),len(outputs.split(" ")), model)
        raw_output = [t.split("[Response_End]")[0] for t in outputs.split("[Response_Start]") if "[Response_End]" in t][0] if "[Response_End]" in outputs else outputs
        
        return raw_output,cost
    def insert_attributions_posthoc_paragraph_all(
        self,
        text: str,
        ctxs: List[str],
        model_name: str = "gpt4o",
        max_tokens: int = 1000
    ) -> str:
        passages = "".join(f"[{i}] {p}\n" for i, p in enumerate(ctxs or []))

        sentences = text.split("\n")
        updated, ph_sentence, prompts = [], {}, []

        for idx, sent in enumerate(sentences):
            if len(sent.strip()) < 10:
                updated.append(sent)
            else:
                key = f"[replace_{idx}]"
                updated.append(key)
                ph_sentence[key] = sent

        for s in ph_sentence.values():
            prompt = instructions.posthoc_attributions_paragraph_all.format_map(
                {"statement": s, "passages": passages}
            )
            prompts.append(prompt)

        client = get_model(model_name)
        outputs: List[str] = []
        cost=0
        for p in prompts:
            raw = client.chat(messages=p, max_tokens=max_tokens).strip()
            cost+= = calculate_openai_api_cost(len(p.split(" ")),len(raw.split(" ")), model)
            if "[Response_End]" in raw:
                processed = [
                    seg.split("[Response_End]")[0]
                    for seg in raw.split("[Response_Start]")
                    if "[Response_End]" in seg
                ][0]
                outputs.append(processed.strip())
            else:
                outputs.append(raw)

        for k, gen in zip(ph_sentence.keys(), outputs):
            ph_sentence[k] = gen if gen else ph_sentence[k]

        final_sents = [ph_sentence.get(tok, tok) for tok in updated]
        return "\n".join(final_sents),cost       
    def run(self, item, model, max_tokens=3000, max_thread=4):
        total_cost = 0
        outline,oulinecost=self.gen_outline(item, model, max_tokens=500)
        total_cost+=oulinecost
        #print("Outline \n",outline)
        initial_res,ini_cost=self.initial_result(item, model,outline, max_tokens)
        total_cost+=ini_cost
        #print("Initial Result \n",initial_res,"\n\n ctxs \n",ctxs)
        if "\n\n References" in initial_res:
            initial_res = initial_res.split("\n\n References")[0]
        refs = []
        for doc in item["ctxs"][:self.top_n]:
            refs.append({
                "title": doc["title"] if "title" in doc and len(doc["title"]) > 0 else "",
                "text": doc["text"]
            })
        root=TreeNode(item["input"],initial_res,True)
        root.set_references(refs)
        init,qs,cost=self.initial_critic(initial_res,item["input"],outline, model,max_tokens=500)
        #print("Init \n",init)
        if init==True:
            print("No further retrieval needed")
            total_cost+=cost
            ref_papers=item["ctxs"]
            return initial_res,total_cost,ref_papers
        else:
            print("Conduct further retrieval")
            total_cost+=cost
            queries = qs

            with ThreadPoolExecutor(max_workers=max_thread) as executor:
                futures = []
                
                for query in queries:
                    child_node = TreeNode(query=query, answer='', state=False)
                    root.add_child(child_node)
                    futures.append(executor.submit(self.gap_critic, child_node, model, 0, 1, 500))
                
                for future in as_completed(futures):
                    res, child_cost = future.result()
                    total_cost += child_cost
                    
            final_answer, final_refs,cost_botup = self.bottom_up_aggregate(outline,root, model, maxtokens=max_tokens)
            total_cost+=cost_botup
            t=0
            while t<1:
                tmp=final_answer
                final_answer,cost_refine=self.reflective_refine(root,final_answer,final_refs,model,outline,max_tokens=5000)
                total_cost+=cost_refine
                if final_answer==tmp:
                    break
                t+=1

            ref_papers = []
            for paper in final_refs:
                    paper_info = {
                        "id": paper.get("paperID") or "",
                        "title": paper.get("title") or "",
                        "text": paper.get("text") or ""
                    }
                    ref_papers.append(paper_info)
            attr_text,costat = self.insert_attributions_posthoc_paragraph_all(
                text=final_answer,
                ctxs=ref_papers,
                model_name=model
            )
            total_cost+=costat
            return attr_text,total_cost,ref_papers
            
price_per_million = {"gpt4o": 2.50,"gpt-4o": 2.50, "gpt-4o-2024-08-06": 2.50, "gpt-4o-2024-05-13": 5.00, "gpt-4o-mini": 0.15, "gpt-4o-mini-2024-07-18": 0.15, "gpt-4-turbo": 10.0, "gpt-3.5-turbo-0125": 0.50} 
price_per_million_output = {"gpt4o": 10.00, "gpt-4o": 10.00, "gpt-4o-2024-08-06": 10.00,  "gpt-4o-2024-05-13": 15.00, "gpt-4o-mini": 0.600, "gpt-4o-mini-2024-07-18": 0.600, "gpt-4-turbo": 30.0, "gpt-3.5-turbo-0125": 1.50} 

def calculate_openai_api_cost(input_tokens: int, output_tokens: int, model_name: str) -> float:
    """
    Calculate OpenAI API cost based on the number of input and output tokens.
    
    Args:
    - input_tokens (int): Number of tokens in the input.
    - output_tokens (int): Estimated number of tokens in the output.
    - price_per_million_tokens (float): Cost per 1 million tokens (e.g., 0.02 for GPT-4).

    Returns:
    - float: The total API cost.
    """
    total_cost_input = (input_tokens / 1000000) * price_per_million[model_name]
    total_cost_output =  (output_tokens / 1000000) * price_per_million_output[model_name]
    total_cost = total_cost_input + total_cost_output
    return round(total_cost, 6)