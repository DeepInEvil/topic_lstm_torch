import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from LSTM_topic import LSTMCell
import torch.nn.init as init
import numpy as np


class LSTMClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, emb_weights, hidden_dim, label_size, batch_size, use_gpu):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        #self.word_embeddings.weight.data = torch.Tensor(emb_weights)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, dropout=0.3, batch_first=False)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim, 500)
        self.fc2 = nn.Linear(500, label_size)

    def init_hidden(self, batch_size):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, sentence):
        self.hidden = self.init_hidden(sentence.size(0))
        x = self.word_embeddings(sentence)
        x = x.view(x.size(1), x.size(0), self.embedding_dim)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out = lstm_out[-1]
        lstm_out = self.fc1(lstm_out)
        out_hidden = self.dropout(lstm_out)
        logits = self.fc2(out_hidden)
        return logits


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        #self.num_layers = num_layers
        #self.embedding_weight = embedding_weight
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = LSTMCell(embed_size, 50, hidden_size, drop=0.2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size, 500)
        self.fc2 = nn.Linear(500, num_classes)
        self.init_weights()

    def init_weights(self):
        #self.embed.weight = nn.Parameter(self.embedding_weight)
        init.xavier_uniform(self.fc1.weight, gain=np.sqrt(2))
        init.xavier_uniform(self.fc2.weight, gain=np.sqrt(2))
        #self.fc1.bias.data.normal_(0, 0.01)
        #self.fc1.weight.data.(0, 0.01)

    def forward(self, x, topic):
        x = self.embed(x)
        #print x.size()
        # Set initial states  & GPU run
        h0 = Variable(torch.zeros(x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(x.size(0), self.hidden_size))
        #print c0.size()
        h0 = (nn.init.xavier_normal(h0))
        c0 = (nn.init.xavier_normal(c0))

        # Forward
        yhat = []
        for j in range(x.size(1)):
            input_t = torch.squeeze(x[:, j: j+1], 1)
            #print input_t.size()
            hx, cx = self.lstm(input_t, topic, (h0, c0))
            #print hx.size()
            yhat.append(hx)
        #print (yhat[-1].size())
        # Decode hidden state of last time step/ many to one


        lstm_out = yhat[-1]
        lstm_out = self.fc1(lstm_out)
        out_hidden = self.dropout(lstm_out)
        logits = self.fc2(out_hidden)
        return logits
