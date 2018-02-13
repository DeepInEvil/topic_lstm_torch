import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
lda_model = '/data/dchaudhu/all_unlabelled/CNN_T_learning/new_dat/lda_models/amazon_lda'


def get_topics(texts, vocab):
    texts = [[vocab.itos[idx] for idx in sent] for sent in texts]
    return texts


def train(train_iter, dev_iter, vocab, model, args):
    if args.cuda:
        model.cuda()
    print model.parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #print vocab.itos[10]
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    print len(train_iter)
    for epoch in range(1, args.epochs+1):
        print epoch
        for iter, traindata in enumerate(train_iter):
            #print train_iter[iter]
            feature, target = traindata.text, traindata.labels
            #print target
            feature.data.t_(), target.data.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            #text_vals = get_topics(feature.data.numpy(), vocab)
            #print text_vals[0]
            #topic_vec = torch.autograd.Variable(torch.from_numpy(np.random.rand(feature.size(0), 50)))
            #print topic_vec.size()
            #topic_vec = topic_vec.type(torch.FloatTensor)
            model.zero_grad()
            #logit = model(feature, topic_vec)
            logit = model(feature)

            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            print iter
            continue
            if iter % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/traindata.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.data[0], 
                                                                             accuracy,
                                                                             corrects,
                                                                        traindata.batch_size))

            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.labels
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        #topic_vec = torch.autograd.Variable(torch.from_numpy(np.random.rand(feature.size(0), 50)))
        #topic_vec = topic_vec.type(torch.FloatTensor)
        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch .max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.data[0][0]+1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
