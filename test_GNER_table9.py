import unittest
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# Load Tokenizers and Models for CPU
tokenizer_llama = AutoTokenizer.from_pretrained("dyyyyyyyy/GNER-LLaMA-7B")
model_llama = AutoModelForCausalLM.from_pretrained("dyyyyyyyy/GNER-LLaMA-7B").to("cpu")

tokenizer_t5 = AutoTokenizer.from_pretrained("dyyyyyyyy/GNER-T5-xxl")
model_t5 = AutoModelForSeq2SeqLM.from_pretrained("dyyyyyyyy/GNER-T5-xxl").to("cpu")

# Function to run prediction and get response for GNER-LLaMA (on CPU)
def get_llama_response(sentence):
    inputs = tokenizer_llama(sentence, return_tensors="pt").to("cpu")  # Changed to CPU
    outputs = model_llama.generate(**inputs, max_new_tokens=640)
    response = tokenizer_llama.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

# Function to run prediction and get response for GNER-T5 (on CPU)
def get_t5_response(sentence):
    inputs = tokenizer_t5(sentence, return_tensors="pt").to("cpu")  # Changed to CPU
    outputs = model_t5.generate(**inputs, max_new_tokens=640)
    response = tokenizer_t5.decode(outputs[0], skip_special_tokens=True).strip()
    return response


# Unit tests for the model predictions
class TestGNERModelPredictions(unittest.TestCase):

    ### GNER-LLaMA Tests (Cases 1-5) ###

    def test_omission_case_1_llama(self):
        raw_sentence = "who directed the film the lorax"
        expected_prediction = "who directed the lorax"
        
        response = get_llama_response(raw_sentence)
        self.assertEqual(response, expected_prediction, "Omission Case 1: Incorrect response from GNER-LLaMA.")

    def test_omission_case_2_llama(self):
        raw_sentence = "any reasonably priced indian restaurants in the theater district"
        expected_prediction = "any reasonably priced indian restaurants in theater district"
        
        response = get_llama_response(raw_sentence)
        self.assertEqual(response, expected_prediction, "Omission Case 2: Incorrect response from GNER-LLaMA.")

    def test_addition_case_3_llama(self):
        raw_sentence = "the conservative regionalist navarra suma finished first and . . ."
        expected_prediction = "the conservative regionalist regionalist navarra suma finished first and . . ."
        
        response = get_llama_response(raw_sentence)
        self.assertEqual(response, expected_prediction, "Addition Case 3: Incorrect response from GNER-LLaMA.")

    def test_substitution_case_4_llama(self):
        raw_sentence = "which five star italian restaurants in manattan have the best reviews"
        expected_prediction = "which five star italian restaurants in manhattan have the best reviews"
        
        response = get_llama_response(raw_sentence)
        self.assertEqual(response, expected_prediction, "Substitution Case 4: Incorrect response from GNER-LLaMA.")

    def test_substitution_case_5_llama(self):
        raw_sentence = "polyethylene terephthalate ( pet ) bottles are made from ethylene and p-xylene ."
        expected_prediction = "polyethylene terephthalate ( p e t) bottles are made from ethylene and p-xylene ."
        
        response = get_llama_response(raw_sentence)
        self.assertEqual(response, expected_prediction, "Substitution Case 5: Incorrect response from GNER-LLaMA.")

    ### GNER-T5 Tests (Cases 6-10) ###

    def test_omission_case_6_t5(self):
        raw_sentence = ". . . whose debut album tol cormpt norz norz norz rock hard journalist wolfrüdiger mühlmann considers a part of war metal ’s roots ."
        expected_prediction = ". . . whose debut album tol cormpt norz norz rock hard journalist wolf-rüdiger mühlmann considers a part of war metal ’s roots ."
        
        response = get_t5_response(raw_sentence)
        self.assertEqual(response, expected_prediction, "Omission Case 6: Incorrect response from GNER-T5.")

    def test_omission_case_7_t5(self):
        raw_sentence = "jennifer lien starred in this action film of the the last six years that received a really good rating"
        expected_prediction = "jennifer lien starred in this action film of the last six years that received a really good rating"
        
        response = get_t5_response(raw_sentence)
        self.assertEqual(response, expected_prediction, "Omission Case 7: Incorrect response from GNER-T5.")

    def test_addition_case_8_t5(self):
        raw_sentence = ". . . performed by wet wet wet that remained at number 1 . . ."
        expected_prediction = ". . . performed by wet wet wet wet that remained at number 1 . . ."
        
        response = get_t5_response(raw_sentence)
        self.assertEqual(response, expected_prediction, "Addition Case 8: Incorrect response from GNER-T5.")

    def test_addition_case_9_t5(self):
        raw_sentence = ". . . liked by many people that starred william forsythe"
        expected_prediction = ". . . liked by many people that starred william forsythe the"
        
        response = get_t5_response(raw_sentence)
        self.assertEqual(response, expected_prediction, "Addition Case 9: Incorrect response from GNER-T5.")

    def test_substitution_case_10_t5(self):
        raw_sentence = "four more children followed : charlotte brontë , ( 1816-1855 ) , branwell brontë ( 1817-1848 ) , emily brontë , ( 1818-1848 ) and anne ( 1820-1849 ) ."
        expected_prediction = "four more children followed : charlotte bront , ( 1816-1855 ) , branwell bront ( 1817-1848 ) , emily bront ( 1818-1848 ) and anne ( 1820-1849 ) ."
        
        response = get_t5_response(raw_sentence)
        self.assertEqual(response, expected_prediction, "Substitution Case 10: Incorrect response from GNER-T5.")


if __name__ == '__main__':
    unittest.main()