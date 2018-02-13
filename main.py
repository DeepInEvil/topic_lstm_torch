#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.data as data
import LSTMmodel
import train
import numpy as np
from gensim.models import word2vec
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate [default: 0.0001]')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=100,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=100, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=7, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
parser.add_argument('-lstm-hidden', default=128, help='lstm hidden dimension')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()
review_dir = '/home/deep/cnn-text-classification-pytorch/topic_lstm_torch/reviews/'
lda_model = '/data/dchaudhu/topic_lstm_torch/lda_models/amazon_lda'
#word_vec_file = '/data/dchaudhu/topic_lstm_torch/word_vectors/word2vec_amazon'
domains = ['electronics', 'books', 'kitchen', 'dvd']


def get_index_to_embeddings_mapping(vocab, word_vecs):
    """
    get word embeddings matrix
    :param vocab:
    :param word_vecs:
    :return:
    """
    sd = 1 / np.sqrt(args.embed_dim)
    embeddings = np.random.normal(0, scale=sd, size=[len(vocab), args.embed_dim])
    embeddings = embeddings.astype(np.float32)
    for word in vocab.stoi.keys():
        try:
            embeddings[vocab.stoi[word]] = word_vecs[word]
        except KeyError:
            # map index to small random vector
            # print "No embedding for word '"  + word + "'"
            #words_not_found.append(word)
            embeddings[vocab.stoi[word]] = np.random.uniform(-0.25, 0.25, 300)
    return embeddings


def get_iterator(dataset, batch_size, train=True,
    shuffle=True, repeat=False):
    dataset_iter = data.Iterator(
        dataset, batch_size=batch_size, device=0,
        train=train, shuffle=shuffle, repeat=repeat,
        sort=False
    )
    return dataset_iter


def get_data(text_field, label_field, domain):
    TEXT = text_field
    LABELS = label_field
    d_review_dir = 'leave_out_' + domain + '/'
    # train, val, test = data.TabularDataset.splits(path, train='train.tsv',
    #                                               validation='val.tsv', test='test.tsv', format='tsv',
    #                                               fields=[('text', TEXT), ('label', LABELS)])
    #train = amazon_reader(TEXT, LABELS, path=path, file_n='train.tsv').examples
    #val = amazon_reader(TEXT, LABELS, path=path, file_n='val.tsv').examples
    #test = amazon_reader(TEXT, LABELS, path=path, file_n='test.tsv').examples
    train = data.TabularDataset(
        path= review_dir + d_review_dir + 'train.tsv', format='tsv', skip_header=True,
        fields=[('text', TEXT),
                ('labels', LABELS)])
    val = data.TabularDataset(
        path= review_dir + d_review_dir + 'val.tsv', format='tsv', skip_header=True,
        fields=[('text', TEXT),
                ('labels', LABELS)])
    print val.size
    test = data.TabularDataset(
        path= review_dir + d_review_dir + 'test.tsv', format='tsv', skip_header=True,
        fields=[('text', TEXT),
                ('labels', LABELS)])
    TEXT.build_vocab(train, val)
    LABELS.build_vocab(train, val)
    print (train[0])
    train_vocab = TEXT.vocab
    print len(val), len(test)
    print type(train)
    #train = torch.utils.data.TensorDataset(train.text, train.labels)

    #train_iter, dev_iter, test_iter = data.Iterator.splits(
    #                            (train, val, test), device = -1,
    #                            batch_sizes=(args.batch_size, len(val), len(test)))
    train_iter = get_iterator(dataset=train, batch_size=args.batch_size, shuffle=True)
    dev_iter = None
    test_iter = None
    return train_iter, dev_iter, test_iter, train_vocab


# load data
#print vocab.itos[18]
# train_iter, dev_iter, test_iter = sst(text_field, label_field, device=-1, repeat=False)


# update args and print

args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


# model
#cnn = model.CNN_Text(args)
"""
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()
"""
lstm = None
# train or predict
if args.predict is not None:
    label = train.predict(args.predict, lstm, text_field, label_field, args.cuda)
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
elif args.test:
    try:
        train.eval(test_iter, lstm, args)
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print()
    try:
        for domain in domains:
            print("\nLoading data for domain..." + domain)
            text_field = data.Field(lower=True)
            label_field = data.Field(sequential=False)
            train_iter, dev_iter, test_iter, vocab = get_data(
                text_field, label_field, domain)
            print vocab.stoi['you']
            args.embed_num = len(text_field.vocab)
            args.class_num = len(label_field.vocab) - 1
            #corpus_wordvec = word2vec.Word2Vec.load(word_vec_file)
            #index_to_vector_map = (get_index_to_embeddings_mapping(vocab, corpus_wordvec))
            index_to_vector_map = None
            lstm = LSTMmodel.LSTMClassifier(vocab_size=len(vocab), embedding_dim=args.embed_dim, emb_weights=index_to_vector_map, hidden_dim=args.lstm_hidden, label_size=args.class_num,
                                            batch_size=args.batch_size, use_gpu=args.cuda)
            train.train(train_iter, dev_iter, vocab, lstm, args)
            train.eval(test_iter, lstm, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')

