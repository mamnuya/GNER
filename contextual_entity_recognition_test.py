# contextual_entity_recognition_test.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import unittest
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("dyyyyyyyy/GNER-LLaMA-7B")
model = AutoModelForCausalLM.from_pretrained("dyyyyyyyy/GNER-LLaMA-7B")

# Define context-based test sentences
sentences = [
    "George Washington was the first president of the United States.",
    "I am flying to Washington for the conference next week.",
]

# Define expected labels
expected_labels = [
    ["B-PERSON", "I-PERSON", "O", "O", "O", "O", "B-LOC", "I-LOC"],
    ["O", "O", "O", "B-LOC", "O", "O", "O"],
]

# Run the test
for sentence, expected in zip(sentences, expected_labels):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)
    print(f"Sentence: {sentence}")
    print(f"Expected Labels: {expected}")
    print(f"Model Output: {outputs.logits.argmax(dim=-1)}")  # Placeholder for model output labels
