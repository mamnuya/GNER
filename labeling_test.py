import spacy

# Load spacy model
nlp = spacy.load("en_core_web_sm")

# Adjusted test cases to match spacy's label output
test_cases = [
    ("Did George Clooney make a musical in the 1980s?", ["O", "B-PERSON", "I-PERSON", "O", "O", "O", "O", "B-DATE", "I-DATE", "O"])
]

# Open a log file to save results
with open("labeling_test_results.log", "w") as log_file:
    for text, expected_labels in test_cases:
        doc = nlp(text)
        labels = [token.ent_iob_ + "-" + token.ent_type_ if token.ent_iob_ != 'O' else 'O' for token in doc]
        log_file.write(f"Input: {text}\n")
        log_file.write(f"Expected Labels: {expected_labels}\n")
        log_file.write(f"Actual Labels: {labels}\n")
        log_file.write(f"Match: {expected_labels == labels}\n\n")
        print(f"Input: {text}")
        print(f"Expected Labels: {expected_labels}")
        print(f"Actual Labels: {labels}")
        print(f"Match: {expected_labels == labels}\n")
