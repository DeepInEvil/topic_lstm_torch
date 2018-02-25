import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


class RNN(nn.Module):

    def __init__(self, vocab_size, embed_size, num_output, hidden_size=64,
                                          num_layers=2, batch_first=True, use_gpu=False, embeddings = None):

        '''
        :param vocab_size: vocab size
        :param embed_size: embedding size
        :param num_output: number of output (classes)
        :param hidden_size: hidden size of rnn module
        :param num_layers: number of layers in rnn module
        :param batch_first: batch first option
        '''

        super(RNN, self).__init__()

        # embedding
        self.embedding_dim = embed_size
        self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        if embeddings is not None:
            self.encoder.weight = nn.Parameter(embeddings)
        self.batch_first = batch_first
        self.drop_en = nn.Dropout(p=0.8)
        self.use_gpu = use_gpu
        self.hidden_dim = hidden_size

        # rnn module
        self.rnn = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.0,
            batch_first=True,
            bidirectional=False
        )
        self.rnncell = nn.LSTMCell(
            input_size=embed_size,
            hidden_size=hidden_size
        )

        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_output)

    def init_hidden(self, batch_size):
        if self.use_gpu:
            hx = Variable(torch.zeros(batch_size, self.hidden_dim).cuda())
            cx = Variable(torch.zeros(batch_size, self.hidden_dim).cuda())
        else:
            hx = Variable(torch.zeros(batch_size, self.hidden_dim))
            cx = Variable(torch.zeros(batch_size, self.hidden_dim))

        return hx, cx

    def forward(self, x, seq_lengths):
        '''
        :param x: (batch, time_step, input_size)
        :return: num_output size
        '''

        x_embed = self.encoder(x)
        x_embed = self.drop_en(x_embed)
        #print x_embed.size()
        #packed_input = pack_padded_sequence(x_embed, seq_lengths.cpu().numpy(), batch_first=self.batch_first)
        #x = x.view(x_embed.size(1), x_embed.size(0), self.embedding_dim)
        # r_out shape (batch, time_step, output_size)
        # None is for initial hidden state

        hx, cx = self.init_hidden(x.size(0))

        yhat = []
        for j in range(x_embed.size(1)):
            input_t = torch.squeeze(x_embed[:, j: j + 1], 1)
            #print input_t.size()
            hx, cx = self.rnncell(input_t, (hx, cx))
            # print hx.size()
            yhat.append(hx)
        #ht, ct = self.rnn(x_embed, self.hidden)
        #print ht.size()
        # use mean of outputs
        #out_rnn, _ = pad_packed_sequence(packed_output, batch_first=True)

        #row_indices = torch.arange(0, x.size(0)).long()
        #col_indices = seq_lengths - 1
        #if next(self.parameters()).is_cuda:
        #    row_indices = row_indices.cuda()
        #    col_indices = col_indices.cuda()

        last_tensor = yhat[-1]
        #last_tensor = ht[-1]
        #print last_tensor.size()
        #fc_input = torch.mean(last_tensor, dim=1)
        #last_tensor = ht[:-1]
        fc_input = self.bn2(last_tensor)
        out = self.fc(fc_input)
        #print out.size()
        return out


class RNN_topic(nn.Module):

    def __init__(self, vocab_size, embed_size, num_output, hidden_size=64,
                                          num_layers=2, batch_first=True, use_gpu=False, embeddings = None):

        '''
        :param vocab_size: vocab size
        :param embed_size: embedding size
        :param num_output: number of output (classes)
        :param hidden_size: hidden size of rnn module
        :param num_layers: number of layers in rnn module
        :param batch_first: batch first option
        '''

        super(RNN_topic, self).__init__()

        # embedding
        self.embedding_dim = embed_size
        self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        if embeddings is not None:
            self.encoder.weight = nn.Parameter(embeddings)
        self.batch_first = batch_first
        self.drop_en = nn.Dropout(p=0.8)
        self.use_gpu = use_gpu
        self.hidden_dim = hidden_size

        # rnn module
        self.rnn = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.5,
            batch_first=True,
            bidirectional=False
        )
        self.rnncell = nn.LSTMCell(
            input_size=embed_size,
            hidden_size=hidden_size
        )

        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_output)

    def init_hidden(self, batch_size):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, x, seq_lengths):
        '''
        :param x: (batch, time_step, input_size)
        :return: num_output size
        '''

        x_embed = self.encoder(x)
        x_embed = self.drop_en(x_embed)
        #print x_embed.size()
        #packed_input = pack_padded_sequence(x_embed, seq_lengths.cpu().numpy(), batch_first=self.batch_first)
        #x = x.view(x_embed.size(1), x_embed.size(0), self.embedding_dim)
        # r_out shape (batch, time_step, output_size)
        # None is for initial hidden state


        self.hidden = self.init_hidden(x.size(0))
        ht, ct = self.rnn(x_embed, self.hidden)
        #print ht.size()
        # use mean of outputs
        #out_rnn, _ = pad_packed_sequence(packed_output, batch_first=True)

        row_indices = torch.arange(0, x.size(0)).long()
        col_indices = seq_lengths - 1
        if next(self.parameters()).is_cuda:
            row_indices = row_indices.cuda()
            col_indices = col_indices.cuda()

        last_tensor = ht[row_indices, col_indices, :]
        #last_tensor = ht[-1]
        #print last_tensor.size()
        #fc_input = torch.mean(last_tensor, dim=1)
        #last_tensor = ht[:-1]
        fc_input = self.bn2(last_tensor)
        out = self.fc(fc_input)
        return out
