#!/usr/bin/env python3
"""
Example script to tokenize '/d--/d' using the o200k_base (GPT-4o) tokenizer
and print the list of token strings.
"""
import tiktoken

def main():
    # Initialize the o200k_base tokenizer (GPT-4o)
    enc = tiktoken.get_encoding("o200k_base")
    text = "/d--/d"
    # Encode the text to token IDs
    tokens = enc.encode(text)
    # Decode each token ID back to bytes, then to string
    token_bytes = enc.decode_tokens_bytes(tokens)
    token_strs = [b.decode("utf-8", errors="replace") for b in token_bytes]
    print(token_strs)

if __name__ == "__main__":
    main()
