from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load the BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")


# Define sentences and their expected labels for entity recognition
test_sentences = [
    {
        "text": "The event was held in NYC last weekend.",
        "expected_labels": ['O', 'O', 'O', 'B-LOC', 'O', 'O']
    },
    {
        "text": "The Prime Minister of the UK made a speech.",
        "expected_labels": ['O', 'B-TITLE', 'O', 'O', 'B-LOC', 'O', 'O']
    },
]

# Run each sentence through the model and print out the results
for sentence in test_sentences:
    tokens = tokenizer.tokenize(sentence["text"])
    inputs = tokenizer(sentence["text"], return_tensors="pt")
    outputs = model(**inputs).logits

    # Get the predicted label for each token
    predicted_labels = torch.argmax(outputs, dim=2).squeeze().tolist()
    
    # Decode the predicted labels
    predicted_tags = [model.config.id2label[label] for label in predicted_labels]

    # Print the tokenized output, expected labels, and model predictions
    print(f"\nSentence: {sentence['text']}")
    print(f"Tokens: {tokens}")
    print(f"Expected Labels: {sentence['expected_labels']}")
    print(f"Predicted Labels: {predicted_tags}")
    print(f"Match: {sentence['expected_labels'] == predicted_tags[:len(sentence['expected_labels'])]}")
