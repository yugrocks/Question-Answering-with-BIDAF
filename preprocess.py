import numpy as np

# First we load the Glove vectors file
vocab_size = 400000


print("loading glove vectors...")
embeddings = np.zeros((vocab_size+1, 50)) # initializing as all zeros
vocab = dict() # a dict of words such that their values contain their indexes
vocab["__pad__"] = 0 # word at index 0 is the padding token


# the index 0 of embedding matrix will contain all zeros for padding
with open("data/glove.6B.50d.txt",encoding='utf8') as file:
    idx = 1
    for line in file:
        values = line.split()
        vocab[values[0]] = idx # add the word in vocab
        embeddings[idx] = np.asarray(values[1:], dtype='float32')
        idx += 1



print("glove vectors loaded")

