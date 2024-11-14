<p align="center"><h2 align="center">Reproduction: Rethinking Negative Instances for Generative Named Entity Recognition</h2></p>

<p align="center">
    <a href="https://github.com/yyDing1/GNER/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/yyDing1/GNER"></a>
    <a href="https://huggingface.co/collections/dyyyyyyyy/gner-65dda2cb96c6e35c814dea56"><img alt="Pretrained Models" src="https://img.shields.io/badge/🤗 HuggingFace-Pretrained Models-green"></a>
    <a href="https://arxiv.org/abs/2402.16602"><img alt="Paper" src="https://img.shields.io/badge/📄-Paper-orange"></a>
    <a href="https://opennlg.cn/"><img src="https://img.shields.io/badge/Organization-OpenNLG%20Group-blueviolet"></a>
</p>

We introduce GNER, a **G**enerative **N**amed **E**ntity **R**ecognition framework, which demonstrates enhanced zero-shot capabilities across unseen entity domains. Experiments on two representative generative models, i.e., LLaMA and Flan-T5, show that the integration of negative instances into the training process yields substantial performance enhancements. The resulting models, GNER-LLaMA and GNER-T5, outperform state-of-the-art (SoTA) approaches by a large margin, achieving improvements of 8 and 11 points in $F_1$ score, respectively. Code and models are publicly available.

* 📖 Paper: [Rethinking Negative Instances for Generative Named Entity Recognition](https://arxiv.org/abs/2402.16602)
* 💾 Models in the 🤗 HuggingFace Hub: [GNER-Models](https://huggingface.co/collections/dyyyyyyyy/gner-65dda2cb96c6e35c814dea56)
* 🧪 Reproduction Materials: [Materials](https://drive.google.com/drive/folders/1m2FqDgItEbSoeUVo-i18AwMvBcNkZD46?usp=drive_link)
* 🎨 Example Jupyter Notebooks: [GNER Notebooks](notebook.ipynb)

<p align="center">
<img src="assets/zero_shot_results.png">
</p>

## PreTrained Models

We release five GNER models based on LLaMA (7B) and Flan-T5 (base, large, xl and xxl).

| Model         | # Params | Zero-shot Average $F_1$ | Supervised Average $F_1$ |          🤗 HuggingFace<br />Download Link          |
| ------------- | -------: | :----------------------: | :-----------------------: | :-------------------------------------------------: |
| GNER-LLaMA    |       7B |           66.1           |           86.09           | [link](https://huggingface.co/dyyyyyyyy/GNER-LLaMA-7B) |
| GNER-T5-base  |     248M |           59.5           |           83.21           | [link](https://huggingface.co/dyyyyyyyy/GNER-T5-base) |
| GNER-T5-large |     783M |           63.5           |           85.45           | [link](https://huggingface.co/dyyyyyyyy/GNER-T5-large) |
| GNER-T5-xl    |       3B |           66.1           |           85.94           |  [link](https://huggingface.co/dyyyyyyyy/GNER-T5-xl)  |
| GNER-T5-xxl   |      11B |           69.1           |           86.15           |  [link](https://huggingface.co/dyyyyyyyy/GNER-T5-xxl)  |

## Demo usage

Please check out [Example Jupyter Notebooks](notebook.ipynb) for guidance on utilizing GNER models.

A simple inference example is as follows:

GNER-LLaMA:

```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM
>>> tokenizer = AutoTokenizer.from_pretrained("dyyyyyyyy/GNER-LLaMA-7B")
>>> model = AutoModelForCausalLM.from_pretrained("dyyyyyyyy/GNER-LLaMA-7B", torch_dtype=torch.bfloat16).cuda()
>>> model = model.eval()
>>> instruction_template = "Please analyze the sentence provided, identifying the type of entity for each word on a token-by-token basis.\nOutput format is: word_1(label_1), word_2(label_2), ...\nWe'll use the BIO-format to label the entities, where:\n1. B- (Begin) indicates the start of a named entity.\n2. I- (Inside) is used for words within a named entity but are not the first word.\n3. O (Outside) denotes words that are not part of a named entity.\n"
>>> sentence = "did george clooney make a musical in the 1980s"
>>> entity_labels = ["genre", "rating", "review", "plot", "song", "average ratings", "director", "character", "trailer", "year", "actor", "title"]
>>> instruction = f"{instruction_template}\nUse the specific entity tags: {', '.join(entity_labels)} and O.\nSentence: {sentence}"
>>> instruction = f"[INST] {instruction} [/INST]"
>>> inputs = tokenizer(instruction, return_tensors="pt").to("cuda")
>>> outputs = model.generate(**inputs, max_new_tokens=640)
>>> response = tokenizer.decode(outputs[0], skip_special_tokens=True)
>>> response = response[response.find("[/INST]") + len("[/INST]"):].strip()
>>> print(response)
"did(O) george(B-actor) clooney(I-actor) make(O) a(O) musical(B-genre) in(O) the(O) 1980s(B-year)"
```

GNER-T5:

```python


>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
>>> tokenizer = AutoTokenizer.from_pretrained("dyyyyyyyy/GNER-T5-xxl")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("dyyyyyyyy/GNER-T5-xxl", torch_dtype=torch.bfloat16).cuda()
>>> model = model.eval()
>>> instruction_template = "Please analyze the sentence provided, identifying the type of entity for each word on a token-by-token basis.\nOutput format is: word_1(label_1), word_2(label_2), ...\nWe'll use the BIO-format to label the entities, where:\n1. B- (Begin) indicates the start of a named entity.\n2. I- (Inside) is used for words within a named entity but are not the first word.\n3. O (Outside) denotes words that are not part of a named entity.\n"
>>> sentence = "did george clooney make a musical in the 1980s"
>>> entity_labels = ["genre", "rating", "review", "plot", "song", "average ratings", "director", "character", "trailer", "year", "actor", "title"]
>>> instruction = f"{instruction_template}\nUse the specific entity tags: {', '.join(entity_labels)} and O.\nSentence: {sentence}"
>>> inputs = tokenizer(instruction, return_tensors="pt").to("cuda")
>>> outputs = model.generate(**inputs, max_new_tokens=640)
>>> response = tokenizer.decode(outputs[0], skip_special_tokens=True)
>>> print(response)
"did(O) george(B-actor) clooney(I-actor) make(O) a(O) musical(B-genre) in(O) the(O) 1980s(B-year)"
```

## Task schema: Incorporating negative instances into training

<p align="center">
<img src="assets/task_schema.png">
</p>

## Hierarchical Matching: A faster algorithm for structuring

We develop a Hierarchical Matching algorithm that provides a straightforward and effective solution to the omission, addition, and substitution problems in the structuring process.

Furthermore, we implement a fast version of the LCS algorithm within $O(N\log N)$, based on the nature of the small number of duplicate words in the query sentence.

First, we transform the Longest Common Subsequence (LCS) problem into a Longest Increasing Subsequence (LIS) problem. Subsequently, we construct a Directed Acyclic Graph (DAG) to facilitate the traceback of the specific sequence.

# A fast version of LCS with a complexity of O(NlogN)
# in the condiction that there are few depulicate words in the sentence
# input: a = [word_1, word_2, ..., word_n], b = [word_1, word_2, ..., word_m]
# return: match_idx = [idx_1, idx_2, ..., idx_n] (correspoding matching index between a and b)
def lcs_solve_fast(a, b):
    n, m = len(a), len(b)
    match_idx = [-1] * n
    match_list_b = defaultdict(list)
  
    # First we can convert the LCS problem into a LIS problem,
    # i.e., LCS(a, b) <=> LIS(index_list)
    for idx, word in enumerate(reversed(b)):
        match_list_b[word].append(m - idx - 1)
    index_list = []
    elem_list = []
    for idx, word in enumerate(a):
        if word in match_list_b:
            index_list.extend(match_list_b[word])
            elem_list.extend([idx] * len(match_list_b[word]))

    # then we compute the longest increasing subsequence of index_list
    # we compute a dag, the edges array store the parent of the node, and path store the results
    father, increasing_seq = [[(-1, -1, -1)]], [-1]
    for i in range(len(index_list)):
        if index_list[i] > increasing_seq[-1]:
            father.append([(len(father[-1]) - 1, i, index_list[i])])
            increasing_seq.append(index_list[i])
        else:
            # binary search
            l, r, query_idx = 0, len(increasing_seq) - 1, -1
            while l <= r:
                mid = (l + r) >> 1
                if increasing_seq[mid] >= index_list[i]:
                    query_idx = mid
                    r = mid - 1
                else:
                    l = mid + 1
            father[query_idx].append((len(father[query_idx - 1]) - 1, i, index_list[i]))
            increasing_seq[query_idx] = index_list[i]

    # finally, we trace back the path to get a solution of the original LCS problem
    i, j = len(father) - 1, len(father[-1]) - 1
    while i > 0:
        match_idx[elem_list[father[i][j][1]]] = father[i][j][2]
        j = father[i][j][0]
        i -= 1
    return match_idx
```

## How to use GNER (updated as of November 2024)

### Requirements

You should install the dependencies:

```bash
# CUDA 11.7 and above
# PyTorch 2.0 and above.
# transformers>=4.32.0,<4.38.0
python -m pip install -r requirements.txt
```

### Quick Reproduction 

We also provide all the generated results for quick reproduction of our results. The `model_predictions` folder contains the generated results of GNER-LLaMA-7B and GNER-T5-xxl (including the ground truth). You can execute the following commands to evaluate the generated results:

```python
# 0shot performance of GNER-LLaMA
python evaluate.py --tokenizer-path yahma/llama-7b-hf --prediction-path model_predictions/llama-7b-task-adaptation-beam1.jsonl
# 0shot performance of GNER-T5-xxl
python evaluate.py --tokenizer-path google/flan-t5-xxl --prediction-path model_predictions/flan-t5-xxl-task-adaptation-beam1.jsonl
```

Other generated results can be found at [here](https://drive.google.com/drive/folders/1kg7YDRk8jK4_Bo19jJpZtdAQMBoucppW?usp=drive_link), and the execution process is similar to the two examples mentioned above.

### Reproduce and Test Paper Results in Table 9 
To verify the results in Table 9 using unit tests, you can executve the following command:
```python
# Table 9 Testing
python test_GNER_table9.py

# Run single tests
python -m unittest test_GNER_table9.TestGNERModelPredictions.test_omission_case_1_llama
python -m unittest test_GNER_table9.TestGNERModelPredictions.test_omission_case_2_llama

python -m unittest test_GNER_table9.TestGNERModelPredictions.test_addition_case_3_llama

python -m unittest test_GNER_table9.TestGNERModelPredictions.test_substitution_case_4_llama
python -m unittest test_GNER_table9.TestGNERModelPredictions.test_substitution_case_5_llama

python -m unittest test_GNER_table9.TestGNERModelPredictions.test_omission_case_6_t5
python -m unittest test_GNER_table9.TestGNERModelPredictions.test_omission_case_7_t5

python -m unittest test_GNER_table9.TestGNERModelPredictions.test_addition_case_8_t5
python -m unittest test_GNER_table9.TestGNERModelPredictions.test_addition_case_9_t5

python -m unittest test_GNER_table9.TestGNERModelPredictions.test_substitution_case_10_t5




```

### Reproduce and Test Paper Results in Figure 6 and Table 10
To optionally observe the outputs of the following tests, view table1case1output.txt and figure6output.txt
```python
#Figure 6 Test
python figure6.py

#Table10 Case 1 Test
python table10case1.py
```

### Perform Robustness Testing 
To optionally observe the outputs of the following tests, view robusttest1.txt
```python
python test_robust1.py
```

### Run provided evaluation script on an empty existing .json file

This produces a ZeroDivisionError.

```python
python evaluate.py --tokenizer-path yahma/llama-7b-hf --prediction-path model_predictions/test_gner_evaluation_empty.jsonl
```

### Run provided evaluation script on existing .json file with limited data and word edits
```python
python evaluate.py --tokenizer-path yahma/llama-7b-hf --prediction-path model_predictions/test_gner_evaluation.jsonl
```


### Training & Inference

First, you should download the training data from [here](https://drive.google.com/drive/folders/1jJsqDhR8Pdg4Qlh5pHm0WZ6Nsk1wEcv9?usp=drive_link), put it in the current directory and rename it as `data`

The training scripts are outlined in folder `scripts`, you can train and evaluate the model by the following command:

```bash
# Train and evaluate LLaMA Model
bash scripts/train_llama_task_adaptation.sh
# Evaluate only
bash scripts/eval_llama_task_adaptation.sh

# Train T5 xxl Model
bash scripts/train_t5_xxl_task_adaptation.sh
# Evaluate only
bash scripts/eval_t5_task_adaptation.sh
```

# Error Analysis
   We performed a detailed error analysis on the GNER models to understand where they may struggle with entity recognition. The following tests and results help highlight the model’s limitations and areas for improvements

Prerequisites
Python 3.x
Install all dependencies using:
```bash
pip install -r requirements.txt
```

## 1. Contextual Entity Recognition Test

This test evaluates the model's ability to recognize entities based on the context in which they appear. It checks whether the model can accurately identify and label entities in sentences with varying structures and complexities.

Implementation Details
Script: contextual_entity_recognition_test.py
Model Used: bert-large-cased-finetuned-conll03-english from Hugging Face Transformers.

Process:
Tokenize input sentences.
Run the model to predict entity labels.
Compare the predicted labels with expected labels.
Determine if the predictions match expectations

How to Run
Execute the following command in your terminal:

```bash
python contextual_entity_recognition_test.py
```
Tests Conducted
Contextual Entity Recognition Test

Objective: To check if the model accurately identifies entities in sentences where the context is essential.

Observation: The model generally recognized entities correctly in straightforward contexts. However, it struggled with ambiguous cases, often mislabeling or failing to recognize entities in complex sentence structures.

Error Found: In cases with multiple interpretations, the model occasionally applied incorrect labels, showing a limitation in contextual understanding.

## 2. Synonym Entity Test

This test assesses the model's ability to recognize entities expressed through synonyms or abbreviations. It evaluates whether the model can generalize entity recognition beyond exact matches to include equivalent terms.

Implementation Details

Script: synonym_entity_test.py
Model Used: bert-large-cased-finetuned-conll03-english
Process:
Provide sentences with entities represented by synonyms or abbreviations.
Predict entity labels using the model.
Compare predictions with expected labels.
How to Run
Execute the following command:
```bash
python synonym_entity_test.py
```
Objective: To evaluate the model’s ability to recognize entities when synonyms or abbreviations are used instead of exact terms.
Observation: The model performed inconsistently with synonyms and abbreviations. For example, it recognized "NYC" as a location but failed with more unusual or varied synonyms for common terms.
Error Found: The model often failed to generalize across equivalent terms, indicating a need for improved synonym and abbreviation handling during training.

## 3. Tokenization Test

This test verifies the model's tokenization accuracy, focusing on complex words like hyphenated terms and multi-token phrases. Proper tokenization is essential for accurate entity recognition.

Implementation Details
Script: tokenization_test.py
Tokenizer Used: bert-base-uncased tokenizer.
Process:
Define test cases with expected tokens.
Tokenize input phrases using the tokenizer.
Compare the tokenizer's output with expected tokens.
Determine if tokenization matches expectations.
How to Run
Run the following command:
```bash
python tokenization_test.py
```
Objective: To assess how well the tokenizer processes complex words, like hyphenated terms or multi-token phrases, which can affect the entity recognition accuracy.
Observation: For most simple terms, tokenization was accurate. However, in cases of hyphenated words and multi-part terms, tokenization was sometimes incorrect, leading to mismatches with expected outputs.
Error Found: Incorrect tokenization on complex words led to misalignment in entity tagging, which could propagate errors in the recognition process.



## 4. Resource Constraint Test

This test monitors the model's memory usage during inference to evaluate its efficiency and suitability for deployment in resource-constrained environments.

Implementation Details
Script: resource_constraint_test.py
Process:
Load a BERT model for sequence classification.
Perform multiple inferences while recording memory usage.
Output memory usage data to a log file for analysis.
How to Run
Execute the script with:
```bash
python resource_constraint_test.py
```

Objective: To monitor the model's memory usage during inference, ensuring it can run efficiently in memory-limited environments.
Observation: Memory usage remained consistent across test cases, but larger sentences led to minor spikes.
Error Found: No critical errors were found, though resource use was high, suggesting potential optimization needs for deployment on devices with restricted memory.

## 5. Labeling Test

This test verifies the model’s ability to accurately label entities in a sentence, comparing the actual labels assigned by the model with the expected labels. It assesses the model’s consistency and accuracy in identifying and labeling entities in diverse sentence structures.

Implementation Details

Script: labeling_test.py
Process:
Load a BERT model fine-tuned for token classification.
Run several test sentences through the model, comparing its output labels with predefined, expected labels.
Output each test sentence with its expected vs. actual labels and note if they match or not.
How to Run
Execute the script with:
```bash
python labeling_test.py
```

Objective: To assess the model's ability to accurately assign entity labels in sentences, ensuring consistency in labeling diverse entities such as names, dates, and locations.

Observation: The model accurately labeled entities in most cases, correctly identifying entities like B-PERSON, B-DATE, and B-LOC. However, minor inconsistencies were observed with more complex sentence structures, which impacted accuracy slightly.

Error Found: No critical errors were found in basic entity labeling, though minor misclassifications occurred in complex sentence structures, indicating a need for fine-tuning on more varied datasets to improve labeling accuracy in edge cases.


# Conclusion

Tokenization: The model performed well with simpler token structures but showed minor inconsistencies with multi-token words and phrases. This suggests that while tokenization accuracy is generally high, special cases may benefit from refinement or tokenization adjustments.

Synonym Entity Detection: Synonym detection worked accurately in identifying common synonyms but faced challenges with less common synonyms or more abstract expressions. This highlights an area where the model could benefit from expanded synonym mapping or additional training data.

Contextual Entity Recognition: The model successfully identified entities within their contextual usage, though certain entity types were misclassified when presented in less straightforward contexts. This indicates the model's strong base performance but suggests further training could improve performance in nuanced contexts.

Resource Constraints: The memory usage test revealed consistent but high memory consumption, particularly for longer sentences, indicating that while the model is stable in resource usage, optimizations would be beneficial for deployment in memory-constrained environments.

Labeling: Entity labeling was generally accurate, with successful tagging of person names, dates, and locations. However, complex sentence structures resulted in minor errors, suggesting the need for improved handling of intricate or nested entities.

Overall Findings: The error analysis demonstrates that the model is reliable in straightforward scenarios but has limitations in handling complex inputs, synonyms, and memory efficiency. The results highlight areas for further enhancement, including synonym handling, memory optimization, and contextual accuracy in entity recognition. These findings suggest that, with targeted refinements, the model's performance and deployability in diverse scenarios could be significantly improved.

## Citation

```bibtex
@misc{ding2024rethinking,
      title={Rethinking Negative Instances for Generative Named Entity Recognition}, 
      author={Yuyang Ding and Juntao Li and Pinzheng Wang and Zecheng Tang and Bowen Yan and Min Zhang},
      year={2024},
      eprint={2402.16602},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
