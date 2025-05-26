import requests
import time
import requests
import os
import instructions
from llms import LanguageModel, get_model

os.environ["S2_API_KEY"]=''
S2_API_KEY=os.environ["S2_API_KEY"]

def search_paper_via_query(query, max_paper_num=5):
    if "Search queries: " in query:
        query = query.split("Search queries: ")[1]
    query_params = {'query': query, 'limit': max_paper_num, "minCitationCount": 5, "sort": "citationCount:desc", 'fields': 'title,year,abstract,authors.name,citationCount,year,url,externalIds'}
    api_key = S2_API_KEY
    # Define headers with API key
    headers = {'x-api-key': api_key}
    # Send the API request
    response = requests.get('https://api.semanticscholar.org/graph/v1/paper/search', params=query_params, headers=headers)
    time.sleep(0.5)

    if response.status_code == 200:
        response_data = response.json()
    # Process and print the response data as needed
    else:
        response_data = None
        print(f"Request failed with status code {response.status_code}: {response.text}")
    # except:
        # response_data = None
    if response_data is None or len(response_data) == 0 or "data" not in response_data:
        print("retrieval failed")
        return None
    else:
        return response_data["data"]
    
def is_integer_string(s):
    return s.isdigit()
    
    
def search_paper_via_query_w_citation_expansion(query, max_paper_num=5, min_citation_count=10):
    if "Search queries: " in query:
        query = query.split("Search queries: ")[1]
    
    query_params = {'query': query, 'limit': max_paper_num, "minCitationCount": 5, "sort": "citationCount:desc", 'fields': 'title,year,abstract,authors.name,citationCount,externalIds'}
    api_key = S2_API_KEY
    headers = {'x-api-key': api_key}
    
    # Send the API request
    response = requests.get('https://api.semanticscholar.org/graph/v1/paper/search', params=query_params, headers=headers)
    time.sleep(0.5)

    if response.status_code == 200:
        response_data = response.json()
    else:
        response_data = None
        print(f"Request failed with status code {response.status_code}: {response.text}")

    if response_data is None or len(response_data) == 0 or "data" not in response_data:
        print("Retrieval failed")
        return None
    else:
        papers = response_data["data"]
        all_papers_data = []  # To store all the papers' data including references and citations
        paper_data_query_params = {'fields': 'title,year,abstract,citationCount,authors,externalIds,isInfluential'}

        for paper in papers:
            paper_data = {
                "paper": paper,
                "references": [],
                "citations": []
            }

            # Query for references (only keep those with citation count >= 10)
            if "paperId" in paper:
                ref_url = f'https://api.semanticscholar.org/graph/v1/paper/{paper["paperId"]}/references'
                ref_response = requests.get(ref_url, params=paper_data_query_params, headers=headers)
                time.sleep(1)  # Respect API rate limit
                if ref_response.status_code == 200:
                    ref_data = ref_response.json()
                    for ref in ref_data.get('data', []):
                        if ref.get("citationCount", 0) >= min_citation_count:
                            paper_data["references"].append(ref)

            # Query for citations (only keep those with citation count >= 10)
            if "paperId" in paper:
                cite_url = f'https://api.semanticscholar.org/graph/v1/paper/{paper["paperId"]}/citations'
                cite_response = requests.get(cite_url, params=paper_data_query_params, headers=headers)
                time.sleep(1)  # Respect API rate limit
                if cite_response.status_code == 200:
                    cite_data = cite_response.json()
                    for cite in cite_data.get('data', []):
                        if cite.get("citationCount", 0) >= min_citation_count:
                            paper_data["citations"].append(cite)

            # Add the paper's data to the total list
            all_papers_data.append(paper_data)

        # Sort papers based on citation count and influence
        sorted_papers = sorted(all_papers_data, key=lambda x: (x['paper'].get('citationCount', 0), x['paper'].get('isInfluential', False)), reverse=True)

        return sorted_papers


def search_snippet_via_query_w_citation_expansion(query, max_paper_num=5, min_citation_count=10):
    if "Search queries: " in query:
        query = query.split("Search queries: ")[1]

    query_params = {
        'query': query,
        'limit': max_paper_num,
        "minCitationCount": min_citation_count,
    }
    
    api_key = S2_API_KEY
    headers = {'x-api-key': api_key}

    response = requests.get('https://api.semanticscholar.org/graph/v1/snippet/search', params=query_params, headers=headers)
    time.sleep(4)  
    
    if response.status_code != 200:
        print(f"Search request failed: {response.status_code}, {response.text}")
        return None, False,0

    response_data = response.json()
    if "data" not in response_data or not response_data["data"]:
        print("No results found.")
        return None, False,0

    papers_snippets = []
    for item in response_data["data"]:
        if "paper" in item and "corpusId" in item["paper"] and "snippet" in item and "text" in item["snippet"]:
            papers_snippets.append({
                "corpusId": item["paper"]["corpusId"],
                "snippet_text": item["snippet"]["text"]
            })

    batch_ids = [f'CorpusID:{p["corpusId"]}' for p in papers_snippets]
    paper_params = {'fields': 'title,abstract,authors.name,citationCount,url,paperId'}
    payload = {
    "ids": batch_ids
    }

    try:
        r = requests.post(
            'https://api.semanticscholar.org/graph/v1/paper/batch',
            params=paper_params,
            json=payload,
            headers=headers
        )
        if r.status_code != 200:
            print(f"Batch request failed with status code: {r.status_code}")
            print("Response:", r.text)
        time.sleep(3)
    except Exception as e:
        print("Error occurred during batch request:", e)
        return None, False,0
    
    all_papers_data = []
    batch_results = r.json()
    for i, paper_info in enumerate(batch_results):
        if not isinstance(paper_info, dict):
            print(f"⚠️ Unexpected item at index {i}, skipping: {paper_info}")
            continue
        snippet_text = papers_snippets[i]["snippet_text"]

        paper_data = {
            "paper": {
                "paperId": paper_info.get("paperId", ""),        
                "title": paper_info.get("title", ""),
                "abstract": paper_info.get("abstract", ""),
                "authors": [a["name"] for a in paper_info.get("authors", [])],
                "citationCount": paper_info.get("citationCount", 0),
                "url": paper_info.get("url", ""),
                "snippet_text": snippet_text
            },
            "references": [],
            "citations": []
        }
        all_papers_data.append(paper_data)
    ctxs = ''
    for entry in all_papers_data:
        p = entry["paper"]  
        paper_text = f"Title: {p['title']}\nAbstract: {p['abstract']}\nRelated snippet: {p['snippet_text']}"
        ctxs += paper_text + "\n\n"

    input_query = instructions.decision_expansion.format_map({"question": query, "snippets": ctxs})
    
    client = get_model("gpt4o")
    output = client.chat(messages=input_query, max_tokens=20)
    cost=calculate_openai_api_cost(len(input_query.split(" ")),len(output.split(" ")), "gpt4o")
    if "[end]terminate" in output:
            return all_papers_data, False, cost
    else:
        for paper_data in all_papers_data:        
            #references
            ref_url = f'https://api.semanticscholar.org/graph/v1/paper/{paper_data["paper"]["paperId"]}/references'
            ref_response = requests.get(ref_url, params={'fields': 'paperId,title,abstract,authors,citationCount,url,isInfluential'}, headers=headers)
            time.sleep(4) 
            
            if ref_response.status_code == 200:
                ref_data = ref_response.json()
                for ref in ref_data.get('data', []):
                    cited_paper = ref.get("citedPaper", {})
                    citations=cited_paper.get("citationCount", 0)
                    if citations is not None and citations >= min_citation_count :
                        paper_data["references"].append({
                            "paperId": cited_paper.get("paperId", ""),
                            "title": cited_paper.get("title", ""),
                            "abstract": cited_paper.get("abstract", ""),
                            "authors": [author["name"] for author in cited_paper.get("authors", [])],
                            "citationCount": cited_paper.get("citationCount", 0),
                            "url": cited_paper.get("url", ""),
                            "isInfluential": ref.get("isInfluential", False)
                        })

            # citations
            cite_url = f'https://api.semanticscholar.org/graph/v1/paper/{paper_data["paper"]["paperId"]}/citations'
            cite_response = requests.get(cite_url, params={'fields': 'paperId,title,abstract,authors,citationCount,url,isInfluential'}, headers=headers)
            time.sleep(4) 

            if cite_response.status_code == 200:
                cite_data = cite_response.json()
                for cite in cite_data.get('data', []):
                    cited_paper = cite.get("citingPaper", {})
                    citations=cited_paper.get("citationCount", 0)
                    if citations is not None and citations >= min_citation_count :
                        paper_data["citations"].append({
                            "paperId": cited_paper.get("paperId", ""),
                            "title": cited_paper.get("title", ""),
                            "abstract": cited_paper.get("abstract", ""),
                            "authors": [author["name"] for author in cited_paper.get("authors", [])],
                            "citationCount": cited_paper.get("citationCount", 0),
                            "url": cited_paper.get("url", ""),
                            "isInfluential": cite.get("isInfluential", False)
                        })
        return all_papers_data, True , cost


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

        
if __name__ == "__main__":
    query = "Limitations for multi-turn LLM Question Answering"
    top_papers,status,cost = search_snippet_via_query_w_citation_expansion(query,1)
    if top_papers is None:
        print(query)
    else:
        if status:
            print(top_papers)
            print("\n\nyes",cost)