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



random.seed(73)
torch.manual_seed(73)

EMBEDDING_DIM = 300
HIDDEN_DIM = 300
EPOCHS = 15
LEARNING_RATE = 0.1
# LAYERS = 2




def prepare_sequence_sent(seq, to_idx):
    idxs=[to_idx['<UNK_WORD>'] if w not in to_idx else to_idx[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def prepare_sequence_tag(seq, to_idx):
    idxs=[to_idx['<UNK_TAG>'] if w not in to_idx else to_idx[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)
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


#get word2idx from word2idx.pickle
import pickle
with open('word2idx.pickle', 'rb') as f:
    word2idx = pickle.load(f)
    
#get tag2idx from tag2idx.pickle
with open('tag2idx.pickle', 'rb') as f:
    tag2idx = pickle.load(f)

#get idx2tag from idx2tag.pickle
with open('idx2tag.pickle', 'rb') as f:
    idx2tag = pickle.load(f)



#load model



model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word2idx), len(tag2idx))
model.load_state_dict(torch.load('model.pt'))
model.eval()


#for taking input sentence from the user and and tokenizing it with NLTK
sentence = input("Enter the sentence: ")
# sentence = "i am a boy"
# print(sentence)
words = word_tokenize(sentence)
words = [word.lower() for word in words ]


predicted = [idx2tag[i] for i in model(prepare_sequence_sent(words, word2idx)).argmax(1).tolist()]

for i in range(len(words)):
    print(words[i],"\t",predicted[i])

