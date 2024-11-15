from transformers import AutoModelForSequenceClassification
import psutil
import time
import unittest
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# Load the BERT model for sequence classification
model = AutoModelForCausalLM.from_pretrained("dyyyyyyyy/GNER-LLaMA-7B")

print("Starting resource-intensive test...")

# Monitor memory usage
with open("resource_usage_test.log", "w") as log_file:
    for _ in range(10):  # Adjust the range as needed for a longer test
        memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MB
        log_file.write(f"Memory usage: {memory_usage:.2f} MB\n")
        print(f"Memory usage: {memory_usage:.2f} MB")
        time.sleep(1)  # Adjust the sleep duration as needed
