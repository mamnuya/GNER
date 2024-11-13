import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Test cases in different languages and with special characters
test_cases = [
    "Did George Clooney make a musical in the 1980s?",  # English
    "¿George Clooney hizo un musical en los años 1980?",  # Spanish
    "জর্জ ক্লুনি কি ১৯৮০ এর দশকে একটি সঙ্গীত তৈরি করেছিলেন?",  # Bangla
    "!*! @ˆ& *( $%$"  # Special characters
]

# Open a log file to save results
with open("robustness_test_results.log", "w") as log_file:
    for text in test_cases:
        doc = nlp(text)
        labels = [token.ent_iob_ + "-" + token.ent_type_ if token.ent_iob_ != 'O' else 'O' for token in doc]
        log_file.write(f"Input: {text}\n")
        log_file.write(f"Labels: {labels}\n\n")
        print(f"Input: {text}")
        print(f"Labels: {labels}\n")


