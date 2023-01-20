# Import necessary libraries
import csv
import openai
import os
import numpy as np
import pandas as pd
import pickle
import re
from transformers import GPT2TokenizerFast

DOC_EMBEDDINGS_MODEL = "text-embedding-ada-002"
QUERY_EMBEDDINGS_MODEL = "text-embedding-ada-002"

# Define a function to parse the text file
def parse_text_file(file_path):
    # Open the file and read its contents
    with open(file_path, 'r') as file:
        contents = file.read()
    
    # Split the contents into sentences or paragraphs
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', contents)
    # sentences = re.split(r'\n', contents)  # for splitting into paragraphs
    
    # Return the array of sentences/paragraphs
    return sentences


def get_embedding(text: str, model: str) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def get_doc_embedding(text: str) -> list[float]:
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)

def get_query_embedding(text: str) -> list[float]:
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

def compute_doc_embeddings(sentences: list[str]) -> list[tuple[str, list[float]]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """

    return [
        (sentence, get_doc_embedding(sentence)) for sentence in sentences
    ]

def write_to_csv(filename: str, data: list[tuple[str, list[float]]]) -> None:
    """
    Write a list of tuples of the form (str, list[float]) to a CSV file.
    """
    with open("embeddings.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for item in data:
            sentence, embedding = item
            writer.writerow([sentence] + embedding)

def read_from_csv(filename: str) -> list[tuple[str, list[float]]]:
    """
    Read a CSV file and return a list of tuples of the form (str, list[float]).
    """
    data = []
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            sentence = row[0]
            embedding = [float(x) for x in row[1:]]
            data.append((sentence, embedding))
    return data

# Call the function with the file path
sentences = parse_text_file('transcriptions.txt')

write_to_csv("embeddings.csv", compute_doc_embeddings(sentences))

ret = read_from_csv("embeddings.csv")
print(ret[100])


