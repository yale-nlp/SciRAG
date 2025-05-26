import argparse
import random
from tqdm import tqdm
import json
import os
import re
import numpy as np
import os
import datasets
import jsonlines
from pipe import pipeline

def load_jsonlines(file):
    try:
        with jsonlines.open(file, 'r') as jsonl_f:
            lst = [obj for obj in jsonl_f]
    except:
        lst = []
        with open(file) as f:
            for line in f:
                lst.append(json.loads(line))
    return lst

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


def process_paragraph(text):
    text = text.replace("<cit.>", "")
    text = remove_citations(text)
    return text

def process_input_data(data, use_contexts=True):
    processed_data = []
    for item in data:
        if "answer" not in item:
            item["answer"] = ""
        if "input" not in item:
            if "question" in item:
                item["input"] = item["question"]
            if "query" in item:
                item["input"] = item["query"]

        new_ctxs = []
        if use_contexts is True:
            # normalize ctx format for different retrieval APIs
            for ctx in item["ctxs"]:
                if type(ctx) is list:
                    for c in ctx:
                        if type(c) is dict:
                            new_ctxs.append(c)
                if type(ctx) is dict:
                    new_ctxs.append(ctx)
            item["ctxs"] = new_ctxs

            # remove duplicated contexts
            processed_paras = []
            for ctx in tqdm(item["ctxs"]):
                if "retrieval text" in ctx:
                    ctx["text"] = ctx["retrieval text"]
                if ctx["text"] is None or len(ctx["text"]) ==0:
                    continue
                if type(ctx["text"]) != str:
                    ctx["text"] = " ".join(ctx["text"]["contexts"])
                ctx["text"] = process_paragraph(ctx["text"])
                if "title" not in ctx:
                    ctx["title"] = ""
                processed_paras.append(ctx)

            processed_paras_dicts = {paper["text"][:100] + paper["title"]: paper for paper in processed_paras}
            processed_paras = list(processed_paras_dicts.values())

            item["ctxs"] = processed_paras
            item["original_ctxs"] = processed_paras
        processed_data.append(item)
    return processed_data



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="path to input file")
    parser.add_argument("--output_file", type=str, help="path to output file")
    parser.add_argument("--dataset", type=str, default=None, help="specify the HF data path if you load them from HF datasets.")
    parser.add_argument("--model_name", type=str, default='gpt4o',help="model name")
    parser.add_argument("--max_tokens", type=int, default=3000)

    args = parser.parse_args()

    # load input data
    if args.input_file is not None:
        if args.input_file.endswith("jsonl"):
            data = load_jsonlines(args.input_file)
        else:
            data = json.load(open(args.input_file))
            if "data" in data:
                data = data["data"]
    elif args.dataset is not None:
        data = list(datasets.load_dataset(args.dataset)["test"])
    else:
        raise ValueError("Please provide either input_file or dataset")
    
    
    final_results = []
    '''
    # Restarting from existing results if there's file whose name matches the output file
    if os.path.isfile(args.output_file):
        final_results = json.load(open(args.output_file))["data"]
        data = data[len(final_results):]
        
        print("restarting from {}".format(len(final_results)))
    '''

    data = process_input_data(data)

    for item in data:
        if "answer" not in item and "output" in item:
            item["answer"] = item["output"]
    
    pipe=pipeline()
        
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_file, 'a', encoding='utf8') as f:
        for idx, item in tqdm(enumerate(data)):
            answer,cost,refs = pipe.run(item=item, model=args.model_name, max_tokens=3000, max_thread=4)
            entry = {
                "query": item["query"],
                "answer_text": answer,
                "cost": cost,
                "ctxs": refs
            }

            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            f.flush()  
            
if __name__ == '__main__':
    main()