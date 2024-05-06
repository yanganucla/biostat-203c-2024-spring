# HW2
# Name: Yang An
# Collaborators: None
# Date: 04/23/2024

import random

def count_characters(s):
    """
    Counts the number of times each character appears in a string.
    
    Args:
    s (str): The input string from which characters will be counted.

    Returns:
    dict: A dictionary with characters as keys and their counts as values.
    """
    char_count = {}
    for i in range(len(s)):
        char = s[i]
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    return char_count

def count_ngrams(s, n=1):
    """
    Counts the number of times each n-gram occurs in a string.
    
    Args:
    s (str): The input string from which n-grams will be counted.
    n (int): The length of the n-grams to be counted.

    Returns:
    dict: A dictionary of n-grams and their counts.
    """
    ngram_count = {}
    for i in range(len(s) - n + 1):
        ngram = s[i:i+n]
        if ngram in ngram_count:
            ngram_count[ngram] += 1
        else:
            ngram_count[ngram] = 1
    return ngram_count

def markov_text(s, n, length=100, seed="Emma Woodhouse"):
    """
    Generates text using an nth-order Markov model based on the input string.

    Args:
        s (str): The source text for the Markov model.
        n (int): The order of the model.
        length (int): The length of text to generate (default 100).
        seed (str): The initial seed string for generating text.

    Returns:
        str: The generated text of length `len(seed) + length`.
    """
    # Compute n+1-gram frequencies
    ngrams = count_ngrams(s, n+1)
    result = seed  # Start with the seed
    
    # Generate text one character at a time
    for _ in range(length):
        current = result[-n:]  # Get the last n characters from the result
        # Find possible extensions
        possibilities = {gram: count for gram, count in ngrams.items() if gram.startswith(current)}
        # If no possible extension, break the loop
        if not possibilities:
            break
        # Select the next character based on weighted probability of the occurrences
        total = sum(possibilities.values())
        choices, weights = zip(*possibilities.items())
        next_char = random.choices(choices, weights=[count/total for count in weights], k=1)[0][-1]
        result += next_char  # Append the selected character to the result

    return result

