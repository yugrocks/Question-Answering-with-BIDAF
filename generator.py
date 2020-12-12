import numpy as np
import pickle









class YugsGenerator:
    
    def __init__(self, embeddings, vocab, X, y, batch_size=32, context_length=300,question_length=50):
        self.X  = X; self.y=y; self.embeddings=embeddings;self.vocab=vocab;self.batch_size=batch_size;self.context_length=context_length
        self.question_length = question_length
        self.__unk__ = np.random.randn(1,50)
        with open("unknown_vector" , "wb") as file:
            pickle.dump(self.__unk__,file)
        self.delete_undesired_data()
            
    
    def delete_undesired_data(self):
        X_new =  [] ; y_new = []
        j = 0
        for i in range(len(self.X)):
            x  = self.X[i]
            if not self.check_if_discard(x):
                X_new.append(self.X[i])
                y_new.append(self.y[i])
            else:
                j += 1
                print("Discarding data")
        print("Total rows discarded = " , j)
    
    def batch_transform(self, X_batch, y_batch):
        # transform this batch into accepted format
        para_vectors = []; question_vectors = []; 
        answer_end = []; answer_begin = []
        for i in range(len(X_batch)):
            # for every element in batch
            para, question = X_batch[i]
            para_vector = self.vectorize_sentence(para, self.context_length)
            question_vector = self.vectorize_sentence(question ,self.question_length)
            para_vectors.append(para_vector)
            question_vectors.append(question_vector)
            # now convert the start and the end to one hot vectors
            ans_start, ans_end = y_batch[i]
            ans_start = self.convert_to_one_hot(ans_start)
            ans_end = self.convert_to_one_hot(ans_end)
            answer_begin.append(ans_start); answer_end.append(ans_end)
        return (np.squeeze(np.array(para_vectors)), np.squeeze(np.array(question_vectors))), (np.array(answer_end), np.array(answer_begin))
            
            
            
    def vectorize_sentence(self, sentence,max_length):
        # sentence is a list of words
        # assuming that length of sentence is always <= max_length
        matrix = []
        for word in sentence:
            if word.lower() in self.vocab:
                vector = self.embeddings[self.vocab[word.lower()]].reshape(50,1)
            else:
                vector = self.__unk__.reshape(50,1)
            matrix.append(vector)
        for i in range(max_length-len(sentence)): # pad sequence
            matrix.append(embeddings[0].reshape(50,1))
        
        return matrix
    
    def check_if_discard(self,x):
        # if length of x is greater than context length 
        if len(x[0]) > self.context_length or len(x[1]) > self.question_length:
            return True
        else:
            return False
    
    def convert_to_one_hot(self, number):
        vec = np.zeros((self.context_length,1))
        vec[number] = 1
        return vec
    
    def get_generator(self):
        index = 0
        while True:
            X = []; y = []
            if index+self.batch_size > len(self.X):
                X.extend(self.X[index:len(self.X)]); X.extend(self.X[0:index+self.batch_size-len(self.X)])
                y.extend(self.y[index:len(self.X)]); y.extend(self.y[0:index+self.batch_size-len(self.X)])
                index = index+self.batch_size-len(self.X)
            else:
                X = self.X[index : index+self.batch_size]
                y = self.y[index : index+self.batch_size]
                index += self.batch_size
            #now process X and y
            X, y = self.batch_transform(X,y)
            yield X, y
            
        
        

    





#g= YugsGenerator(embeddings, vocab, X,y)
#gen = g.get_generator()

#d = gen.__next__()










