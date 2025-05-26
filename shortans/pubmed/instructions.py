Outline_Generation_Instruction = (
"Given a yes/no question, carefully analyze the information needed to make an accurate judgment. "
"Focus on clarifying the question's key concept, assessing expected supporting evidence, and reasoning towards a simple yes or no conclusion. "
"Based on this analysis, create a concise outline to guide the final response. Specify the sections and the proportion each should contribute to the final answer, totaling 100%.\n"
"Your outline should be marked as [Response_Start] Answer [Response_End].\n"
"Here's an example outline:\n"
"Question: Does vitamin D supplementation reduce the risk of respiratory infections?\n"
"Answer: [Response_Start]\n"
"1. (20%) Restate the question and explicitly provide a judgment in the form of 'Judgment: Yes.' or 'Judgment: No.'\n"
"2. (60%) Briefly summarize key evidence supporting or refuting the answer.\n"
"3. (20%) Note any critical limitations or remaining uncertainties, and reaffirm the final judgment.\n"
"[Response_End]\n"
"Now, please create an outline for this question: {question}"
)



initial_generation_w_references_zero_shot = (
"Provide a detailed verification of the following yes/no question. Structure your response clearly according to the outline below, with each section proportionate to the outline's recommended percentage. Each section must be backed by scientific references, clearly cited.\n"
"Start by clearly stating your judgment in the format 'Judgment: Yes.' or 'Judgment: No.'\n"
"Then briefly explain how you will organize the evidence, followed by a detailed presentation of supporting or contradicting evidence.\n"
"Focus on synthesizing the evidence rather than summarizing each reference separately. Highlight the relationships and conflicts between different studies or pieces of evidence."
"Ensure your answer is coherent, well-organized, and sufficiently detailed so that real-world scientists can confidently evaluate the claim's accuracy. Cite relevant references at the end of each claim-supporting or contradicting statement (e.g., 'Previous studies said xxx, which support this result [1].')."
"Include citations explicitly from the provided references. If multiple sources support or contradict the statement, select and cite the most directly relevant one."
"References: \n{context}"
"\nClaim: {input}"
"\nOutline: {outline}"
"Your answer should be marked as [Response_Start] Answer [Response_End]."
)

Gap_Critic_Prompt_initial = (
    "Please review the following answer based on the outline guidance and the original query. Do you think the current answer fulfills the requirements?"
    "If it provides sufficient information and judgement, return '[end]terminate' in lowercase."
    "Otherwise, provide careful feedback on what additional information is needed to supplement the current answer, and create a new semantic retrieval query (or queries) to gather additional information that can address these gaps."
    "Please note that if these queries have similarities, or focus on similar problem directions or information, they can be merged. Try to ensure that each query given explores different sub-problems. If there is no need to split into multiple queries, then one query is enough."
    "Please give as few queries as possible."
    "Ignore content related to future work directions, conclusions, or acknowledgments. "
    "Please ensure that each query is clear, concise, and contain necessary context or keywords to guide the retrieval process effectively."
    "Do not mention or reference specific sections or content from the current answer in your query. "
    "Please return your queries in the format below:"
    "(1) Your query content."
    "(2) Additional query content if needed."
    "..."
    "Here is the outline : {guidance}\n"
    "Here is the current answer for your review: {answer}"
    "Here is the original query: {query}"
)


Gap_Critic_Prompt_later = (
    "You are in a retrieval chain that has been expanded to better answer the initial core query. The retrieval path is: {path}."
    "Currently, you are at the retrieval step for: {query}."
    "Please review the following answer based on the query. If the answer can roughly answer the query, return '[end]terminate' in lowercase. "
    "Otherwise, provide careful feedback on what additional information is needed to supplement the current answer, and create a new semantic retrieval query (or queries) to gather additional information that can address these gaps."
    "Please note that if these queries have similarities, or focus on similar problem directions or information, they can be merged. Try to ensure that each query given explores different sub-problems. If there is no need to split into multiple queries, then one query is enough."
    "Please give as few queries as possible."
    "Ignore content related to future work directions, conclusions, or acknowledgments. "
    "Please ensure that each query is clear, concise, and contain necessary context or keywords to guide the retrieval process effectively."
    "Do not mention or reference specific sections or content from the current answer in your query. "
    "Please return your queries in the format below:"
    "(1) Your query content."
    "(2) Additional query content if needed."
    "..."
    "Here is the current answer for your review: {answer}"
)


decision_expansion=(
    "Given a scientific question and a set of candidate supporting snippets, do you think the snippets are roughly enough for a basic answer to the question?"
    "Or do you think we need to start from the source paper of these snippets and check its related citations and cited literature to jointly understand and analyze in order to get the correct answer?"
    "If you think the snippets are roughly enough, return '[end]terminate' in lowercase."
    "Otherwise, return [Expansion] in lowercase."
    "Here is the question: {question}"
    "Here are the snippets: {snippets}"
)


additional_generation_w_references_zero_shot = (
"You are in a retrieval chain that has been expanded to better answer the initial research-related core query."
"The retrieval path is: {path}."
"Currently, you are at the retrieval step for: {query}."
"Provide a detailed and informative answer only to the query at current step. Your response should offer a concrete answer."
"Make sure your answer includes summaries of relevant literature or texts or clear descriptions of their contribution to the query. When you make a claim, it is always best to have excerpts or citations to support them."
"Ensure your anwer is well-supported by references. Focus on giving a concrete answer to the query, rather than providing a short or surface-level response."
"Ensure the answer is well-structured, coherent and informative so that real-world scientists can gain a clear understanding of about the query. Rather than simply summarizing multiple papers one by one, try to organize your answers based on similarities and differences between papers." 
"Make sure to add citations to all citation-worthy statements using the provided references (References). More specifically, add the citation number at the end of each relevant sentence e.g., 'This work shows the effectiveness of problem X [1].' when the passage [1] in References provides full support for the statement."
"Not all references may be relevant.You can read through the rationales and think on your own, and only cite those that directly support the statement."
"If multiple references support a statement, cite them together (e.g., [1][2]). Yet, for each citation-worthy statement, you only need to add at least one citation, so if multiple eviences support the statement, just add the most relevant citation to the sentence."
"References: \n{context}"
"Your answer should be marked as [Response_Start] Answer [Response_End]."
)


final_prompt_zero_shot=(
"Please review the following research-related question and its initial answer and read them carefully." 
"The initial answer may have some shortcomings, so we performed additional searches and supplemented information. Now please combine the information from the supplemented query and answer to fill in more content to the original answer, offering a comprehensive overview and clearly structured in multiple paragraphs."
"Organize your answer according to the key themes or sections identified in the outline below, and please note that the length of each part of the answer should be roughly the same as the percentage in the outline. Ensure each section is well-supported by multiple references, not just a single source. "
"Focus on giving a comprehensive overview of the topic, rather than providing a short or surface-level response."
"Ensure the answer is well-structured, coherent and informative so that real-world scientists can gain a clear understanding of the subject. Rather than simply summarizing multiple papers one by one, try to organize your answers based on similarities and differences between papers." 
"Make sure your answer includes summaries of relevant literature or texts or clear descriptions of their contribution to the query. When you make a claim, it is always best to have excerpts or citations to support them."
"Make sure to add citations to all citation-worthy statements using the provided references (References). More specifically, add the citation number at the end of each relevant sentence e.g., 'This work shows the effectiveness of problem X [1].' when the passage [1] in References provides full support for the statement."
"Not all references may be relevant, so only cite those that directly support the statement."
"If multiple references support a statement, cite them together (e.g., [1][2]). Yet, for each citation-worthy statement, you only need to add at least one citation, so if multiple eviences support the statement, just add the most relevant citation to the sentence."
"And update the citation indexes in the answer according to the latest reference list below."
"Make sure your final answer meets the outline requirements."
"\nHere is the initial query: {query}"
"\nHere is the initial answer: {answer}" 
"\nHere is the supplemented queries and answers: {supplement}"
"\nReferences: {context}"   
"\nOutline: {outline}"
"\n"
"Your answer should be marked as [Response_Start] Answer [Response_End]."
)


ref_feedback="""
Given an answer to a scientific query or requirements based on the most recent scientific literature, give me your feedback. 
Ensure the answer is well-structured, coherent and informative so that real-world scientists can gain a clear understanding of the subject. Do not simply summarize multiple papers one by one, but you do should include proper summaries of papers, and try to organize your answers based on similarities and differences between papers.
Make sure your answer includes summaries of relevant literature or texts or clear descriptions of their contribution to the query. When you make a claim, it is always best to have excerpts and citations to support them.
Regarding the content improvements, it is often helpful to ask for more concrete results, applications, or methodologies to different tasks, elaborate on details of crucial methods, or suggest including explicit excerpts and citations as supports.
Stylistic improvements can include better organizations or writing enhancements. 
Your response should be marked as [Response_Start] and [Response_End].
If you think the current answer basically meets all the requirements and has no obvious room for improvement, and can be used as a candidate for a good answer, then return Feedback: [terminate] in lower case.
Else. prioritize the feedback by listing the most critical improvements first. 
Each feedback should be preceded by 'Feedback: '. 
The answer should be organized according to the outline below.
##\n
Question: {question}\n
Answer: {answer}\n
Outline: {outline}\n
[Response_Start]Feedback: [Response_End]
Now, please generate feedback for this question.
##\n
"""


edit_feedback="""
You have been given a research-related question or request, an initial comprehensive answer, and some feedback pointing out possible improvements. 
Now, please refine the answer according to the following guidelines: 

1. **Focus and Organization**: 
   - Provide a thorough, multi-paragraph response, following the key themes or sections identified in the outline. 
   - Ensure that the approximate length and level of detail for each section is consistent with the proportions indicated in the outline, but don't include the percentage of the proportion in your answer.
   - Rather than merely listing studies one by one, organize the discussion based on similarities or differences among the referenced works.

2. **References and Citations**: 
   - Use references from the 'References' section to support all citation-worthy statements, adding their citation number at the end of the sentence, e.g., '[1]'. 
   - If multiple references directly support the same statement, you may group them like '[1][2]'. 
   - Only cite references that truly support the claim, and ensure you re-index citations to match the final reference list if needed. 
   - Do not introduce references that are irrelevant to the statements being made.

3. **Clarity and Comprehensiveness**: 
   - Incorporate feedback to clarify or expand on crucial details, methods, or results. 
   - Strive for a more comprehensive overview of the topic rather than a surface-level summary. 
   - When making a claim or stating an important finding, it is best to briefly illustrate or quote relevant points from the supporting references.

4. **Feedback Integration**: 
   - Only modify parts of the original answer where the feedback indicates improvements are needed,keeping the other sentences unchanged. (e.g., to correct inaccuracies, add clarifications, or reorganize content). 
   - Do not omit any crucial information from the original answer unless the feedback explicitly states that certain sentences are incorrect or redundant and should be removed. 
   - If you add new paragraphs, ensure you are not duplicating content already present in the original response.

5. **Stylistic Consistency**: 
   - Keep the original paragraphs and new lines intact unless the feedback requires changes in structure. 
   - Maintain a coherent narrative flow, with smooth transitions between sections. 
   - Use clear, professional language that real-world scientists would find understandable and informative.

6. **Final Formatting**: 
   - Your refined answer must be enclosed between '[Response_Start]' and '[Response_End]'. 
   - Make sure the final version is well-structured, balanced according to the outline, and thoroughly addresses the question.

Below are the materials you have to work with:

- **Question**: {question}

- **Original Answer**: {original_answer}

- **Feedback**: {feedback}

- **Outline**: {outline}

- **References**: {references}

Following these instructions, please refine the answer accordingly.
Your final answer should be marked between [Response_Start] and [Response_End].

"""


branch_prompt_zero_shot=(
"You are in a retrieval chain that has been expanded to better answer the initial research-related core query."
"The retrieval path is: {path}."
"Currently, you are at the retrieval step for query: {query}."
"Please review the current research-related query and its initial answer and read them carefully." 
"The initial answer may have some shortcomings, so we performed additional searches and supplemented information. Now please combine the information from the supplemented query and answer to optimize the original answer, offering a comprehensive overview and clearly structured in multiple paragraphs."
"Also you should try to keep the original answer content's structure unchanged."
"Ensure the answer is well-structured, coherent and informative so that real-world scientists can gain a clear understanding of the subject, rather than providing a short or surface-level response."
"And re-cite the citations in the answer according to the latest reference list below."
"Make sure your answer includes summaries of relevant literature or texts or clear descriptions of their contribution to the query. When you make a claim, it is always best to have excerpts or citations to support them."
"Make sure to add citations to all citation-worthy statements using the provided references (References). More specifically, add the citation number at the end of each relevant sentence e.g., 'This work shows the effectiveness of problem X [1].' when the passage [1] in References provides full support for the statement."
"Not all references may be relevant.You can read through the rationales and think on your own, and only cite those that directly support the statement."
"If multiple references support a statement, cite them together (e.g., [1][2]). Yet, for each citation-worthy statement, you only need to add at least one citation, so if multiple eviences support the statement, just add the most relevant citation to the sentence."
"Here is the initial answer: {answer}" 
"Here is the supplemented queries and answers: {supplement}"
"Here is the references: {context}"   
"\n"
"Your answer should be marked as [Response_Start] Answer [Response_End]."
)

aggregate_prompt_zero_shot=(
"Please review the following research-related question and its initial answer and read them carefully." 
"The initial answer may have some shortcomings, so we performed additional searches and supplemented information. Now please combine the information from the supplemented query and answer to optimize the original answer, offering a comprehensive overview and clearly structured in multiple paragraphs. "
"Ensure the answer is well-structured, coherent and informative so that real-world scientists can gain a clear understanding of the subject, rather than providing a short or surface-level response."
"Make sure your final answer meets the outline requirements."
"Here is the initial query: {query}"
"Here is the initial answer: {answer}" 
"Here is the outline for your review: {guidance}"
"Here is the supplemented queries and answers: {supplement}"   
"\n"
"Your answer should be marked as [Response_Start] Answer [Response_End]."
)

posthoc_attributions_paragraph_all = """
We give you a short paragraph extracted from an answer to a question related to the most recent scientific literature, and a set of evidence passages.
Find all of the citation-worthy statements without any citations, and insert citation numbers to the statements that are fully supported by any of the provided citations in listed as References. 
If none of the passages support the statement, do not insert any citation, and leave the original sentence as is, but do your best to insert citation. 
If multiple passages provide sufficient support for the statement, you only need to insert one citation, rather than inserting all of them. Your answer should be marked as [Response_Start] and [Response_End].'\n
Here's an example:\n
Statement: Language models store rich knowledge in their parameters during pre-training, resulting in their strong performance on many knowledge-intensive tasks. However, such parametric knowledge based generations are often hard to attribute. Models can also struggle in long-tail knowledge. On the other hand, non-parametric knowledge is retrieved from an external source, such as a large-scale collection of documents, during inference. 
References:
[0] Title: Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models Text: Roberts et al. (2020) shows that T5 (Raffel et al., 2020) can perform a new task formulation, closedbook QA. Concretely, T5 can produce answers to questions without access to any corpus at inference time, instead producing answers based on its model parameters, tuned to remember information digested in pretraining.\n
[1] Title: Reliable, Adaptable, and Attributable Language Models with Retrieval Text: Unlike parametric LMs—which use large-scale text data only during training; retrieval-augmented LMs leverage an external large-scale collection of documents (datastore) at inference by selecting relevant documents from the datastore (Asai et al., 2023a). Retrieval-augmented LMs can W1: largely reduce factual errors (Mallen et al., 2023), W2: provide better attributions (Gao et al., 2023a), W3: enabling flexible opt-in and out of sequences (Min et al., 2024).
[2] Title: Atlas: Few-shot Learning with Retrieval Augmented Language Models Text: In this work we present Atlas, a carefully designed and pre-trained retrieval augmented language model able to learn knowledge intensive tasks with very few training examples. We perform evaluations on a wide range of tasks, including MMLU, KILT and NaturalQuestions, and study the impact of the content of the document index, showing that it can easily be updated. Notably, Atlas reaches over 42\% accuracy on Natural Questions using only 64 examples, outperforming a 540B parameters model by 3\% despite having 50x fewer parameters.
[3] Title: Language Models are Few-Shot Learners Text: Similarly, GPT-3 achieves 64.3% accuracy on TriviaQA in the zero-shot setting, 68.0\% in the one-shot setting, and 71.2\% in the few-shot setting, the last of which is state-of-the-art relative to fine-tuned models operating in the same closed-book setting.
[4] Title: When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories Text:  On both datasets, LMs' memorization (RQ1) is often limited to the popular factual knowledge and even GPT-3 davinci-003 fails to answer the majority of the long-tail questions. Moreover, on such questions, scaling up models does not significantly improve the performance. This also suggests that we can predict if LMs memorize certain knowledge based on the information presented in the input question only. We next investigate whether a semi-parametric approach that augments LMs with retrieved evidence can mitigate the low performance on questions about less popular entities (RQ2). Nonparametric memories largely improve performance on long-tail distributions across models.
[5] Title: Democratizing Large Language Models via Personalized Parameter-Efficient Fine-tuning Text: Personalization in large language models (LLMs) is increasingly important, aiming to align LLM's interactions, content, and recommendations with individual user preferences. Recent advances in LLM personalization have spotlighted effective prompt design, by enriching user queries with non-parametric knowledge through behavior history retrieval and textual profiles. However, these approaches were limited due to a lack of model ownership, resulting in constrained customization and privacy issues. Moreover, they often failed to accurately capture user behavior patterns, especially in cases where user data were complex and dynamic. To address these shortcomings, we introduce One PEFT Per User (OPPU), which employs personalized parameter-efficient fine-tuning (PEFT) modules, to store user-specific behavior patterns and preferences.
[6] Title: RECOMP: Improving Retrieval-Augmented LMs with Context Compression and Selective Augmentation Text:  Retrieval-augmented language models (RALMs) (Khandelwal et al., 2019; Izacard et al., 2022; Lewis et al., 2020; Borgeaud et al., 2022) have shown impressive performance on knowledge-intensive tasks (Kwiatkowski et al., 2019; Petroni et al., 2021). Simply prepending retrieved documents to the input without updating the language models (LMs) (Shi et al., 2023b; Ram et al., 2023; Si et al., 2022) allows retrieval augmentation even for black-box LMs, but such approach comes with limitations. First, it increases computational costs as LMs now encode substantially more tokens. Second, even if we manage to adapt LMs to efficiently incorporate longer context (Beltagy et al., 2020; Zaheer et al., 2020), these models struggle to use all information in the context, frequently missing information placed in the middle (Liu et al., 2023). Third, prepending a large number of documents in-context can further confuse LMs with irrelevant information, degrading model performances (Mallen et al., 2022; Shi et al., 2023a).
[Response_Start]Language models store rich knowledge in their parameters during pre-training, resulting in their strong performance on many knowledge-intensive tasks [3]. However, such parametric knowledge based generations are often hard to attribute [0]. Models can also struggle in long-tail knowledge [4]. On the other hand, non-parametric knowledge is retrieved from an external source, such as a large-scale collection of documents, during inference [2].[Response_End]\n
Now, please insert citations to the following sentece. ##\n
Statement: {statement}
References:\n{passages}\n
"""

system_reasoning = (
"You are in a retrieval chain that has been expanded to better answer the initial core query."
"The retrieval path is: {path}."
"Currently, you are at the retrieval step for: {query}."

"You have a set of partial paper texts (abstracts or snippets). "
"Your goal is to analyze each text's contribution and the relationship between them, build symbolic relationships," 
"and decide which texts are most relevant and contributing to the query and the overall chain."
)


step1_reason_prompt = (
    "We have the following candidate texts from different papers (abstracts or snippets): {paper_text} "
    "The query is: {query_text}\n\n"
    "**Step 1 Task**:\n"
    "1. For each paper, identify its key content segments and label them with:\n"
    "   - T (theoretical part: theorem, definitions, main theoretical results),\n"
    "   - E (experimental part: methodology, experiment details, results),\n"
    "   - A (applications),\n"
    "   - or other labels if needed (e.g., 'M' for methodology if it's not purely experimental).\n"
    "2. For each segment, provide a brief summary (1-2 sentences) \n"
    "   and assess its relevance to the query as High, Medium, or Low.\n\n"
    "**Output format** (example):\n"
    "{{\n"
    '  "papers": [\n'
    "    {{\n"
    '      "paper_index": 1,\n'
    '      "segments": [\n'
    "        {{\n"
    '          "label": "T",\n'
    '          "description": "...",\n'
    '          "relevance": "High/Medium/Low"\n'
    "        }},\n"
    "        {{\n"
    '          "label": "E",\n'
    '          "description": "...",\n'
    '          "relevance": "..."\n'
    "        }}\n"
    "      ]\n"
    "    }},\n"
    "    ...\n"
    "  ]\n"
    "}}\n"
    "Please keep the output structure strictly without additional comments."
)


step2_reason_prompt = (
    "Below is the structured breakdown of each paper's segments from Step 1:\n\n"
    "{step1_result_json}\n\n"
    "Using that breakdown, please establish symbolic relationships among the papers and the query: {query} \n"
    "   - T (theoretical part: theorem, definitions, main theoretical results),\n"
    "   - E (experimental part: methodology, experiment details, results),\n"
    "   - A (applications),\n"
    "   - or other labels if needed (e.g., 'M' for methodology if it's not purely experimental).\n"
    "For example:\n"
    '- "[1]T -> [2]T" means paper1\'s theoretical part informs or extends paper2\'s theoretical part.\n'
    '- "[1]E -> [Q]" means paper1\'s experiment part contributes directly to answering the query.\n'
    '- "[3]A -> [2]T" means paper3\'s application part provides insights for paper2\'s theory, etc.\n\n'
    "or other labels if needed (e.g., 'M' for methodology if it's not purely experimental)."
    "In each relationship, use the format: \"[paper_index][label] -> [paper_index or Q][label (if paper)]\"\n"
    "If the second target is the query itself, just use [Q].\n\n"
    "**Output format** (example):\n"
    "{{\n"
    '  "relationships": [\n'
    "    {{\n"
    '      "symbol": "[1]T -> [Q]",\n'
    '      "rationale": "Paper1\'s theoretical result directly addresses the phenomenon in the query."\n'
    "    }},\n"
    "    {{\n"
    '      "symbol": "[2]E -> [3]T",\n'
    '      "rationale": "Paper2\'s experiment suggests data that confirms the theorem in Paper3."\n'
    "    }},\n"
    "    ...\n"
    "  ]\n"
    "}}\n"
    "Please keep the rationale concise, and keep the output structure strictly without additional comments."
)


step3_reason_prompt = (
    "We now have the symbolic relationships from Step 2:\n\n"
    "{step2_relationships_json}\n\n"
    "Where we have the symbols:"
    "   - T (theoretical part: theorem, definitions, main theoretical results),\n"
    "   - E (experimental part: methodology, experiment details, results),\n"
    "   - A (applications),\n"
    "   - or other labels if needed (e.g., 'M' for methodology if it's not purely experimental).\n"
    "And the breakdown of each paper from Step 1:\n\n"
    "{step1_result_json}\n\n"
    "**Step 3 Task**:\n"
    "1. Decide, at the paper level, which papers are essential for answering the query "
    '"{query}" within the context of the overall retrieval chain "{path}".\n'
    "   (You may consider the segments ([1]T, [1]E, etc.) internally, but your final output "
    "   should only include paper indexes.)\n"
    "2. Provide a final ranked list (most to least relevant) of the retained papers (just the paper_index).\n"
    "3. Discard any paper that is not necessary. If discarding, briefly explain why.\n"
    '4. Clarify the relationships between papers or paper and the query to justify your decisions.\n\n'
    "**Output format** (example):\n"
    "{{\n"
    '  "final_selection": [\n'
    "    {{\n"
    '      "paper_index": 1,\n'
    '      "rank": 1,\n'
    '      "justification": "..."\n'
    "    }},\n"
    "    {{\n"
    '      "paper_index": 3,\n'
    '      "rank": 2,\n'
    '      "justification": "..."\n'
    "    }}\n"
    "    ...\n"
    "  ],\n"
    '  "discarded_items": [\n'
    "    {{\n"
    '      "paper_index": 2,\n'
    '      "reason": "Not relevant to the query"\n'
    "    }},\n"
    "    ...\n"
    "  ]\n"
    "}}\n"
    "Please return the final result in valid JSON and keep the output structure strictly without additional comments.\n"
)


