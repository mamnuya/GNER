import unittest
import json
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import nltk
from evaluate import extract_predictions, parser

# Download required NLTK tokenizer data
nltk.download('punkt')
nltk.download('punkt_tab')

class TestGNERModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load model and tokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained("dyyyyyyyy/GNER-LLaMA-7B")
        cls.model = AutoModelForCausalLM.from_pretrained("dyyyyyyyy/GNER-LLaMA-7B", torch_dtype=torch.bfloat16)
        cls.entity_labels = ["genre", "rating", "review", "plot", "song", "average ratings", "director", "character", "trailer", "year", "actor", "title"]
        cls.instruction_template = (
            "Please analyze the sentence provided, identifying the type of entity for each word on a token-by-token basis.\n"
            "Output format is: word_1(label_1), word_2(label_2), ...\n"
            "We'll use the BIO-format to label the entities, where:\n"
            "1. B- (Begin) indicates the start of a named entity.\n"
            "2. I- (Inside) is used for words within a named entity but are not the first word.\n"
            "3. O (Outside) denotes words that are not part of a named entity.\n"
        )

    def generate_model_output(self, sentence, instruction):
        input_texts = f"[INST] {instruction} [/INST]"
        inputs = self.tokenizer(input_texts, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=640)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[response.find("[/INST]") + len("[/INST]"):].strip()

    def test_bio_output_english(self):
        # English sentence setup
        sentence = "Did George Clooney make a musical in the 1980s?"
        words = nltk.word_tokenize(sentence)
        instruction = (
            f"{self.instruction_template}\nUse the specific entity tags: {', '.join(self.entity_labels)} and O.\n"
            f"Sentence: {' '.join(words)}"
        )

        # Expected outputs
        expected_bio_format = ['O', 'B-actor', 'I-actor', 'O', 'O', 'B-genre', 'O', 'O', 'B-year']

        # Generate model response
        response = self.generate_model_output(sentence, instruction)
        
        # Bio-format prediction extraction
        bio_predictions = extract_predictions({
            "label_list": self.entity_labels,
            "instance": {"words": words},
            "prediction": response,
        })
        self.assertEqual(bio_predictions, expected_bio_format, "BIO-format output does not match expected format for English input.")

    def test_bio_output_spanish(self):
        # Spanish sentence setup
        sentence = "¿George Clooney hizo un musical en los años 1980?"
        words = nltk.word_tokenize(sentence)
        instruction = (
            f"{self.instruction_template}\nUse the specific entity tags: {', '.join(self.entity_labels)} and O.\n"
            f"Sentence: {' '.join(words)}"
        )

        # Expected outputs
        expected_bio_format = ['B-actor', 'I-actor', 'O', 'O', 'B-genre', 'O', 'O', 'O', 'B-year']

        # Generate model response
        response = self.generate_model_output(sentence, instruction)

        # Bio-format prediction extraction
        bio_predictions = extract_predictions({
            "label_list": self.entity_labels,
            "instance": {"words": words},
            "prediction": response,
        })
        self.assertEqual(bio_predictions, expected_bio_format, "BIO-format output does not match expected format for Spanish input.")
        
    def test_bio_output_specialChars(self):
        # Special Character sentence setup
        sentence = "!*! @^& *( *( $%$"
        words = nltk.word_tokenize(sentence)
        instruction = (
            f"{self.instruction_template}\nUse the specific entity tags: {', '.join(self.entity_labels)} and O.\n"
            f"Sentence: {' '.join(words)}"
        )

        # Expected outputs
        expected_bio_format = ['O', 'O', 'O', 'O', 'O']

        # Generate model response
        response = self.generate_model_output(sentence, instruction)

        # Bio-format prediction extraction
        bio_predictions = extract_predictions({
            "label_list": self.entity_labels,
            "instance": {"words": words},
            "prediction": response,
        })
        self.assertEqual(bio_predictions, expected_bio_format, "BIO-format output does not match expected format for Special Char input.")
        
    def test_bio_output_bengali(self):
        # Bengali sentence setup
        sentence = "জর্জ ক্লুনি কি ১৯৮০ সালের দশকে কোনো মিউজিকাল বানিয়েছিলেন?"
        words = nltk.word_tokenize(sentence)
        instruction = (
            f"{self.instruction_template}\nUse the specific entity tags: {', '.join(self.entity_labels)} and O.\n"
            f"Sentence: {' '.join(words)}"
        )

        # Expected outputs
        expected_bio_format = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

        # Generate model response
        response = self.generate_model_output(sentence, instruction)

        # Bio-format prediction extraction
        bio_predictions = extract_predictions({
            "label_list": self.entity_labels,
            "instance": {"words": words},
            "prediction": response,
        })
        self.assertEqual(bio_predictions, expected_bio_format, "BIO-format output does not match expected format for Bengali input.")

if __name__ == "__main__":
    unittest.main()
