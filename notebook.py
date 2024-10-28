#!/usr/bin/env python
# coding: utf-8

# ## Import Requirements

# In[1]:


import nltk
import torch
nltk.download('punkt_tab')
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM


# ## Demo Input and Instructions

# In[2]:


# Demo input
sentence = "did george clooney make a musical in the 1980s"
words = nltk.word_tokenize(sentence)
entity_labels = ["genre", "rating", "review", "plot", "song", "average ratings", "director", "character", "trailer", "year", "actor", "title"]
# print(demo_words)

# fit in the instruction template
instruction_template = "Please analyze the sentence provided, identifying the type of entity for each word on a token-by-token basis.\nOutput format is: word_1(label_1), word_2(label_2), ...\nWe'll use the BIO-format to label the entities, where:\n1. B- (Begin) indicates the start of a named entity.\n2. I- (Inside) is used for words within a named entity but are not the first word.\n3. O (Outside) denotes words that are not part of a named entity.\n"
instruction = f"{instruction_template}\nUse the specific entity tags: {', '.join(entity_labels)} and O.\nSentence: {' '.join(words)}"
print(f"Final Instruction:\n\n{instruction}")


# ## Load Model & Generate

# In[3]:


# For GNER-LLaMA Model
tokenizer = AutoTokenizer.from_pretrained("dyyyyyyyy/GNER-LLaMA-7B")
model = AutoModelForCausalLM.from_pretrained("dyyyyyyyy/GNER-LLaMA-7B", torch_dtype=torch.bfloat16).cuda()
## For LLaMA Model, instruction part are wrapped with [INST] tag
input_texts = f"[INST] {instruction} [/INST]"
inputs = tokenizer(input_texts, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=640)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = response[response.find("[/INST]") + len("[/INST]"):].strip()

# For GNER-T5 Model
# tokenizer = AutoTokenizer.from_pretrained("dyyyyyyyy/GNER-T5-xxl")
# model = AutoModelForSeq2SeqLM.from_pretrained("dyyyyyyyy/GNER-T5-xxl", torch_dtype=torch.bfloat16).cuda()
# inputs = tokenizer(instruction, return_tensors="pt").to("cuda")
# outputs = model.generate(**inputs, max_new_tokens=640)
# response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# In[4]:


print(f"Model Generation: \n\n{response}")


# ## Structure

# In[5]:


# structure the generated text
from evaluate import extract_predictions, parser
example = {
    "label_list": entity_labels,
    "instance": {"words": words},
    "prediction": response,
}


# In[6]:


# bio-format prediction
bio_predictions = extract_predictions(example)
print(f"Predictions (BIO-format): \n\n{bio_predictions}")


# In[7]:


# entity-level prediction
entity_level_predictions = parser(words, bio_predictions)
print(f"Predictions (Entity-level): \n\n{entity_level_predictions}")


# In[8]:


# json-format prediction
import json
from collections import defaultdict
json_dict = defaultdict(list)
for item in entity_level_predictions:
    json_dict[item[1]].append(item[0])
print(f"Predictions (Json-format): \n\n{json.dumps(json_dict, indent=4)}")

