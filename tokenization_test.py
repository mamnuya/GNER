from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("dyyyyyyyy/GNER-LLaMA-7B")

# Define test cases with complex tokenization challenges
inputs = [
    ("Who is directing the Hobbit?", ["who", "is", "directing", "the", "ho", "##bb", "##it", "?"]),
    ("Analyzing multi-token words like re-directing", ["analyzing", "multi", "-", "token", "words", "like", "re", "-", "directing"]),
    ("Analyzing multi-token words like 'high-profile'", ["analyzing", "multi", "-", "token", "words", "like", "high", "-", "profile"]),
    ("Identifying end-to-end encryption methods", ["identifying", "end", "-", "to", "-", "end", "encryption", "methods"]),
]

# Test each input
for input_text, expected_tokens in inputs:
    tokenized_output = tokenizer.tokenize(input_text)
    match = tokenized_output == expected_tokens

    print(f"Input: {input_text}")
    print(f"Expected Tokens: {expected_tokens}")
    print(f"Tokenized Output: {tokenized_output}")
    print(f"Match: {match}\n")
