import json
import numpy as np
from nltk.tokenize import word_tokenize


file = open(r"data/train-v1.1.json", "r")

test_file = open("data/dev-v1.1.json", "r")



data = json.loads(file.read())["data"]
test_data = json.loads(test_file.read())["data"]



X = []
y =  []



def get_word_index(context, start_index, end_index):
    start_index = len(word_tokenize(context[0:start_index-1]))
    end_index = len(word_tokenize(context[0:end_index-1])) -1 
    return start_index, end_index


j = 0
for article in data:
    j += 1
    print(j)
    for paragraph in article["paragraphs"]:
        context = paragraph["context"]
        for qas in paragraph["qas"]:
            question = qas["question"]
            answers = qas["answers"]
            answer = answers[0]
            answer_start = answer["answer_start"] # integer for character index
            text = answer["text"]
            answer_end = len(text)+answer_start-1 # integer
            X.append([word_tokenize(context), word_tokenize(question)])
            start, end = get_word_index(context,answer_start,answer_end )
            y.append([start,end])
            
            
