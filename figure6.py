import unittest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import spacy

class TestSelfCorrectionWithBeamSearch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize tokenizer and model
        cls.tokenizer = AutoTokenizer.from_pretrained("dyyyyyyyy/GNER-LLaMA-7B")
        cls.model = AutoModelForCausalLM.from_pretrained(
            "dyyyyyyyy/GNER-LLaMA-7B", 
            load_in_4bit=True,    
            device_map="auto"     
        )

        # Load spaCy model for NER
        cls.nlp = spacy.load("en_core_web_sm")

        # Sample input
        cls.input_sentence = "What was the fog rated ?"
        cls.ground_truth = [
            "What(O)", "was(O)", "the(B-title)", "fog(I-title)", "rated(O)", "?(O)"
        ]

        # Set up the input tensor
        cls.inputs = cls.tokenizer(cls.input_sentence, return_tensors="pt").to("cpu")

    @classmethod
    def beam_search(cls, input_ids, beam_width=2, max_length=20):
        beams = [(input_ids, 0)]  # (sequence, score)

        for _ in range(max_length):
            new_beams = []

            for seq, score in beams:
                with torch.no_grad():
                    outputs = cls.model(seq)
                    logits = outputs.logits[:, -1, :]

                probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()
                top_probs, top_indices = torch.topk(probs, beam_width)

                for prob, index in zip(top_probs, top_indices):
                    new_seq = torch.cat([seq, index.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score + prob.item()
                    new_beams.append((new_seq, new_score))

            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]  # Keep top beam_width beams

        best_sequence = beams[0][0]
        generated_ids = best_sequence[0].tolist()

        predicted_labels = cls.decode_generated_ids(generated_ids)

        return predicted_labels

    @classmethod
    def decode_generated_ids(cls, generated_ids):
        predicted_labels = []
        current_token = ''

        for token_id in generated_ids:
            token = cls.tokenizer.decode([token_id]).strip()

            # Skip special tokens
            if token in ['<s>', '</s>', '<pad>', '<|endoftext|>', '', ' ']:
                continue
            
            # Concatenate tokens if it's a sub-token
            if token.startswith('##'):
                current_token += token[2:]  # Remove '##' and concatenate
            else:
                if current_token:  # Add the previously accumulated token
                    label = cls.get_label(current_token)
                    predicted_labels.append(f"{current_token}({label})")
                current_token = token  # Start a new token

        # Assign label for the last token if it exists
        if current_token:
            label = cls.get_label(current_token)
            predicted_labels.append(f"{current_token}({label})")


        return predicted_labels

    @classmethod
    def get_label(cls, token):
        """Uses spaCy for labeling based on token content."""
        doc = cls.nlp(token)
        if doc.ents:
            # If the token is recognized as an entity, return the label
            return doc.ents[0].label_  # Return the first entity label found
        else:
            return "O"  # Default label for other tokens

    def test_highest_beam_score(self):
        input_ids = self.tokenizer.encode(self.input_sentence, return_tensors='pt').to(self.model.device)
        predicted_labels = self.beam_search(input_ids, beam_width=2, max_length=20)

        expected_highest_beam = ["What(O)", "was(O)", "the(O)", "fog(O)"]
        self.assertEqual(predicted_labels[:4], expected_highest_beam, 
                         "Highest beam score prediction is incorrect.")

    def test_second_highest_beam_score(self):
        input_ids = self.tokenizer.encode(self.input_sentence, return_tensors='pt').to(self.model.device)
        predicted_labels = self.beam_search(input_ids, beam_width=2, max_length=20)

        expected_second_highest_beam = ["What(O)", "was(O)", "the(B-title)", "fog(I-title)"]
        self.assertEqual(predicted_labels[:4], expected_second_highest_beam, 
                         "Second-highest beam score prediction is incorrect.")

    def test_final_prediction(self):
        input_ids = self.tokenizer.encode(self.input_sentence, return_tensors='pt').to(self.model.device)
        predicted_labels = self.beam_search(input_ids, beam_width=2, max_length=20)

        expected_final_prediction = [
            "What(O)", "was(O)", "the(B-title)", "fog(I-title)", "rated(O)", "?(O)"
        ]
        self.assertEqual(predicted_labels, expected_final_prediction, 
                         "Final prediction after self-correction is incorrect.")
    
    def test_ground_truth_comparison(self):
        input_ids = self.tokenizer.encode(self.input_sentence, return_tensors='pt').to(self.model.device)
        predicted_labels = self.beam_search(input_ids, beam_width=2, max_length=20)

        self.assertEqual(predicted_labels, self.ground_truth, 
                         "Final prediction does not match the ground truth.")

if __name__ == '__main__':
    unittest.main()