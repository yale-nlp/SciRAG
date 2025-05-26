# Code for SciRAG

We provide the implementation of SciRAG for both long-form answers (ScholarQA, QASA) and short-form answers (SciFact, PubMedQA).
The pipelines are largely the same, with only minor differences in input, output, and prompt formats.

You can check and run the pipeline via:

```bash
conda env create -f environment.yml
conda activate scirag
./run.sh
```

For the initial retrieval from the 45 million paper index, we recommend following the method in:
[https://github.com/AkariAsai/OpenScholar/tree/main/retriever](https://github.com/AkariAsai/OpenScholar/tree/main/retriever)
This helps reduce the heavy resource cost during retrieval.
Once retrieved, you can use the output file as the input to the SciRAG pipeline.

