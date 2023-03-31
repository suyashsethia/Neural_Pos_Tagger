from conllu import parse
# import nltk
from nltk.tokenize import word_tokenize

from torchtext.vocab import GloVe
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
# from torchtext.data import Field, BucketIterator
import spacy
import numpy as np
import random
import math
import time

import sklearn
from sklearn.metrics import classification_report

# use to_categorical from keras
# from tensorflow.keras.utils import to_categorical

# random.seed(73)
torch.manual_seed(73)


#train data
data = open('UD_English-Atis/en_atis-ud-train.conllu' ,'r',encoding='utf-8')
data = data.read()

#get the data in the form of a list of sentences containing only the 1st 2nd and 4th columns

data = parse(data)
data = [[(word['id'],word['form'],word['upostag']) for word in sentence] for sentence in data]

# training data
#create a list of sentences which contains a list of words in each sentence and a list of tags in each sentence
training_data = []
for sentence in data:
    words = []
    tags = []
    for word in sentence:
        words.append(word[1].lower())
        tags.append(word[2])
    training_data.append((words,tags))

# print(training_data[0][1])



#validation data
val_data = open('UD_English-Atis/en_atis-ud-dev.conllu' ,'r',encoding='utf-8')
val_data = val_data.read()

#get the data in the form of a list of sentences containing only the 1st 2nd and 4th columns
val_data = parse(val_data)
val_data = [[(word['id'],word['form'],word['upostag']) for word in sentence] for sentence in val_data]

validation_data = []
for sentence in val_data:
    words = []
    tags = []
    for word in sentence:
        words.append(word[1].lower())
        tags.append(word[2])
    validation_data.append((words,tags))




#test data
test_data = open('UD_English-Atis/en_atis-ud-test.conllu' ,'r',encoding='utf-8')
test_data = test_data.read()

#get the data in the form of a list of sentences containing only the 1st 2nd and 4th columns
test_data = parse(test_data)
test_data = [[(word['id'],word['form'],word['upostag']) for word in sentence] for sentence in test_data]

testing_data = []
for sentence in test_data:
    words = []
    tags = []
    for word in sentence:
        words.append(word[1].lower())
        tags.append(word[2])
    testing_data.append((words,tags))



#dividing data into X and Y
X = [[word[1] for word in sentence] for sentence in data]
Y = [[word[2] for word in sentence] for sentence in data]
# print(len(X) , len(Y))
#list of all the words in the training data
words_training = []
for sentence in X:
    for word in sentence:
        words_training.append(word)
# print(len(words_training))

tags_training = []
for sentence in Y:
    for tag in sentence:
        tags_training.append(tag)
# print(len(tags_training))

#dividing validation data into X and Y
val_X = [[word[1] for word in sentence] for sentence in val_data]
val_Y = [[word[2] for word in sentence] for sentence in val_data]
# print(len(val_X) , len(val_Y))
words_validation = []
for sentence in val_X:
    for word in sentence:
        words_validation.append(word)

tags_validation = []
for sentence in val_Y:
    for tag in sentence:
        tags_validation.append(tag)



#dividing test data into X and Y
test_X = [[word[1] for word in sentence] for sentence in test_data]
test_Y = [[word[2] for word in sentence] for sentence in test_data]
# print(len(test_X) , len(test_Y))

words_testing = []
for sentence in test_X:
    for word in sentence:
        words_testing.append(word)

tags_testing = []
for sentence in test_Y:
    for tag in sentence:
        tags_testing.append(tag)


#creating a set of all the unique words in the data , and create a dictionary mapping each word to a unique index
unique_words = set([word for sentence in X for word in sentence])
word2idx = {word:idx for idx,word in enumerate(unique_words)}
idx2word = {idx:word for idx,word in enumerate(unique_words)}

#adding the unknown word to the dictionary at the end
unique_words.add('<UNK_WORD>')
word2idx['<UNK_WORD>'] = len(word2idx)
idx2word[len(idx2word)] = '<UNK_WORD>'


#geting an list of all the unique words in the data
unique_words_list = list(unique_words)
# print(unique_words_list[0])
# print(word2idx['the'])

#create a set of all the unique tags in the data , and create a dictionary mapping each tag to a unique index
unique_tags = set([tag for sentence in Y for tag in sentence])
tag2idx = {tag:idx for idx,tag in enumerate(unique_tags)}
idx2tag = {idx:tag for idx,tag in enumerate(unique_tags)}

#add the unknown tag to the dictionary at the end
tag2idx['<UNK_TAG>'] = len(tag2idx)
idx2tag[len(idx2tag)] = '<UNK_TAG>'
unique_tags.add('<UNK_TAG>')


# print(tag2idx)

#Hyperparameters
EMBEDDING_DIM = 300
HIDDEN_DIM = 300
EPOCHS = 15
LEARNING_RATE = 0.1
LAYERS = 2

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word2idx), len(tag2idx))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)


def prepare_sequence_sent(seq, to_idx):
    idxs=[to_idx['<UNK_WORD>'] if w not in to_idx else to_idx[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def prepare_sequence_tag(seq, to_idx):
    idxs=[to_idx['<UNK_TAG>'] if w not in to_idx else to_idx[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def train_model(model, data, num_epoch):
    
    for epoch in range(num_epoch):    
        for sentence, tags in data:
            model.zero_grad() # clear the gradients of all optimized variables
            model.hidden = model.init_hidden() # initialize the hidden state
            sentence_in = prepare_sequence_sent(sentence, word2idx) # convert the sentence to a tensor
            targets = prepare_sequence_tag(tags, tag2idx) # convert the tags to a tensor
            tag_scores = model(sentence_in) # forward pass
            loss = loss_function(tag_scores, targets) # calculate the loss
            loss.backward() # backward pass
            optimizer.step() # update the parameters
        
        
        print("Epoch:", epoch+1, "Loss:", loss.item())
            
    


train_model(model, training_data, EPOCHS)



#save the model
torch.save(model.state_dict(), 'model.pt')


#load the model
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word2idx), len(tag2idx))
model.load_state_dict(torch.load('model.pt'))
model.eval()


def evaluate(model, data):
    model.eval()
    correct = 0
    total = 0
    for sentence, tags in data:
        sentence_in = prepare_sequence_sent(sentence, word2idx)
        targets = prepare_sequence_tag(tags, tag2idx)
        tag_scores = model(sentence_in)
        _, predicted = torch.max(tag_scores, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    return (100 * correct / total), 


print("accuracy on train set",evaluate(model,training_data))
print("accuracy on test set",evaluate(model,testing_data))
print("accuracy on validation set",evaluate(model,validation_data))


#for training
# sklearn.metrics.classification_report(Y, y_pred, *, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')[source]

#save word2index and tag2index and idx2tag
import pickle
with open('word2idx.pickle', 'wb') as handle:
    pickle.dump(word2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('tag2idx.pickle', 'wb') as handle:
    pickle.dump(tag2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('idx2tag.pickle', 'wb') as handle:
    pickle.dump(idx2tag, handle, protocol=pickle.HIGHEST_PROTOCOL)


#print in a file

#on training 
predicted_training_tags = [idx2tag[i] for i in model(prepare_sequence_sent( words_training,word2idx)).argmax(1).tolist()]
answer_training = sklearn.metrics.classification_report(tags_training, predicted_training_tags)
print("answer_training",answer_training)
#store the 

#on validation
predicted_validation_tags = [idx2tag[i] for i in model(prepare_sequence_sent( words_validation,word2idx)).argmax(1).tolist()]
answer_validation = sklearn.metrics.classification_report(tags_validation, predicted_validation_tags)
print("answer_validation",answer_validation)

#on testing
predicted_testing_tags = [idx2tag[i] for i in model(prepare_sequence_sent( words_testing,word2idx)).argmax(1).tolist()]
answer_testing = sklearn.metrics.classification_report(tags_testing, predicted_testing_tags)
print("answer_testing",answer_testing)


with open('answers.txt', 'w') as f:
    f.write("Hyperparameters used are:")
    f.write("EMBEDDING_DIM",EMBEDDING_DIM)
    f.write("HIDDEN_DIM",HIDDEN_DIM)
    f.write("EPOCHS",EPOCHS)
    f.write("LEARNING_RATE",LEARNING_RATE)
    
    f.write("accuracy on train set",evaluate(model,training_data))
    f.write("accuracy on train set",evaluate(model,training_data))
    f.write("accuracy on train set",evaluate(model,training_data))
    
    f.write("answer_training",answer_training)
    f.write("answer_validation",answer_validation)
    f.write("answer_testing",answer_testing)







