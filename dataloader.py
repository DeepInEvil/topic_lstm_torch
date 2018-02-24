import torch
import pandas as pd
import numpy as np
from collections import Counter

import util as ut

class TextClassDataLoader(object):

    def __init__(self, path_file, word_to_index, batch_size=32):

        self.batch_size = batch_size
        self.word_to_index = word_to_index

        df = pd.read_csv(path_file,delimiter=',')
        df['review'] = df['review'].apply(ut._tokenize)
        df['review'] = df['review'].apply(self.generate_indexifyer())
        #print df
        #df['labels'] = df['labels'].apply(self.get_labels())
        self.samples = df.values.tolist()
        self.shuffle_indices()
        self.n_batches = int(len(self.samples) / self.batch_size)
        self.max_length = self.get_max_length()
        self.report()

    def shuffle_indices(self):
        self.indices = np.random.permutation(len(self.samples))
        self.index = 0
        self.batch_index = 0

    def get_max_length(self):
        length = 0
        for sample in self.samples:
            length = max(length, len(sample[1]))
        return length

    def generate_indexifyer(self):

        def indexify(lst_text):
            indices = []
            for word in lst_text:
                if word in self.word_to_index:
                    indices.append(self.word_to_index[word])
                else:
                    indices.append(self.word_to_index['__UNK__'])
            return indices

        return indexify

    @staticmethod
    def _padding(batch_x):
        batch_s = sorted(batch_x, key=lambda x: len(x))
        size = len(batch_s[-1])
        for i, x in enumerate(batch_x):
            missing = size - len(x)
            batch_x[i] =  batch_x[i] + [0 for _ in range(missing)]
        return batch_x

    def _create_batch(self):
        batch = []
        n = 0
        while n < self.batch_size:
            _index = self.indices[self.index]
            batch.append(self.samples[_index])
            self.index += 1
            n += 1
        self.batch_index += 1

        string, label = tuple(zip(*batch))
        #print label
        label = [1 if lbl == 'positive' else 0 for lbl in label]

        # get the length of each seq in your batch
        seq_lengths = torch.LongTensor(map(len, string))

        # dump padding everywhere, and place seqs on the left.
        # NOTE: you only need a tensor as big as your longest sequence
        seq_tensor = torch.zeros((len(string), seq_lengths.max())).long()
        for idx, (seq, seqlen) in enumerate(zip(string, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

        # SORT YOUR TENSORS BY LENGTH!
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        # seq_tensor = seq_tensor.transpose(0, 1)

        label = torch.LongTensor(label)
        label = label[perm_idx]

        return seq_tensor, label, seq_lengths

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self.shuffle_indices()
        for i in range(self.n_batches):
            if self.batch_index == self.n_batches:
                raise StopIteration()
            yield self._create_batch()

    def show_samples(self, n=10):
        for sample in self.samples[:n]:
            print sample

    def report(self):
        print '# samples: {}'.format(len(self.samples))
        print 'max len: {}'.format(self.max_length)
        print '# vocab: {}'.format(len(self.word_to_index))
        print '# batches: {} (batch_size = {})'.format(self.n_batches, self.batch_size)