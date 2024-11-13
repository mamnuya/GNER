from transformers import AutoTokenizer

# Initialize the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Updated test cases to match BERT's tokenization
test_cases = [
    ("Who is directing the Hobbit?", ["who", "is", "directing", "the", "ho", "##bb", "##it", "?"]),
    ("Analyzing multi-token words like re-directing", ["analyzing", "multi", "-", "token", "words", "like", "re", "-", "directing"])
]

# Open a log file to save results
with open("tokenization_test_results.log", "w") as log_file:
    for sentence, expected_tokens in test_cases:
        tokens = tokenizer.tokenize(sentence)
        # Log results
        log_file.write(f"Input: {sentence}\n")
        log_file.write(f"Expected Tokens: {expected_tokens}\n")
        log_file.write(f"Tokenized Output: {tokens}\n")
        log_file.write(f"Match: {tokens == expected_tokens}\n\n")
        # Print results
        print(f"Input: {sentence}")
        print(f"Expected Tokens: {expected_tokens}")
        print(f"Tokenized Output: {tokens}")
        print(f"Match: {tokens == expected_tokens}\n")

