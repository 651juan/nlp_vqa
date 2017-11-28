# coding: utf-8

"""
Deep CBOW (with minibatching)

Based on Graham Neubig's DyNet code examples:
  https://github.com/neubig/nn4nlp2017-code
  http://phontron.com/class/nn4nlp2017/

"""

import gzip
import json
import parameters
import random
import time
from collections import defaultdict
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(1)
random.seed(1)


CUDA = torch.cuda.is_available()
print("CUDA: %s" % CUDA)


# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]
PAD = w2i["<pad>"]

# One data point
Example = namedtuple("Example", ["words", "tag", "w"])

def read_dataset(questions_path, annotations_path):
    with gzip.GzipFile(questions_path, 'r') as file:
        questions = json.loads(file.read())

    with gzip.GzipFile(annotations_path, 'r') as file:
        annotations = json.loads(file.read())

    for line in range(len(questions['questions'])):
        words = questions['questions'][line]['question'].lower().strip()
        tag = annotations['annotations'][line]['multiple_choice_answer']
        yield Example(words=[w2i[x] for x in words.split(" ")],
                      tag=t2i[tag],
                      w=tag)


# Read in the data
train = list(read_dataset( "data/vqa_questions_train.gzip", "data/vqa_annotatons_train.gzip"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("data/vqa_questions_valid.gzip", "data/vqa_annotatons_valid.gzip"))
nwords = len(w2i)
ntags = len(t2i)

class DeepCBOW(nn.Module):
    """
    Deep CBOW model
    """

    def __init__(self, vocab_size, embedding_dim, output_dim, hidden_dims=[], transformations=[]):
        """
        :param vocab_size: Vocabulary size of the training set.
        :param embedding_dim: The word embedding dimension.
        :param output_dim: The output dimension, ie the number of classes.
        :param hidden_dims: A list of hidden layer sizes. Default: []
        :param transformations: A list of transformation functions.
        """
        super(DeepCBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_num = len(hidden_dims)
        self.linears = {}
        if (self.hidden_num == 0):
            self.linear1 = nn.Linear(embedding_dim, output_dim)
            self.linears[0] = self.linear1
        else:
            self.linear1 = nn.Linear(embedding_dim, hidden_dims[0])
            self.linears[0] = self.linear1
            for i in range(1, self.hidden_num):
                l = "self.linear" + str(i+1)
                exec(l + " = nn.Linear(hidden_dims[i-1], hidden_dims[i])")
                exec("self.linears[i] = " + l)
            l = "self.linear" + str(self.hidden_num + 1)
            exec(l + " = nn.Linear(hidden_dims[self.hidden_num-1], output_dim)")
            exec("self.linears[self.hidden_num] = " + l)
        self.F = transformations

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        h = torch.sum(embeds, 1)
        if(self.hidden_num == 0):
            h = self.linears[0](h)
        else:
            for i in range(self.hidden_num):
                name = self.F[i]
                matrix = self.linears[i](h)
                h = self.func_map(name, matrix)
            h = self.linears[self.hidden_num](h)
        return h

    def func_map(self, name, matrix):
        if name == "relu":
            return F.relu(matrix)
        else:
            # Default
            return F.relu(matrix)

model = DeepCBOW(nwords,
                 parameters.embedding_dim,
                 ntags,
                 parameters.hidden_dims,
                 parameters.transformations)

if CUDA:
    model.cuda()

print(model)

def minibatch(data, batch_size=32):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]


def evaluate(model, data):
    """Evaluate a model on a data set."""
    correct = 0.0

    for batch in minibatch(data):

        seqs, tags = preprocess(batch)
        scores = model(get_variable(seqs))
        _, predictions = torch.max(scores.data, 1)
        targets = get_variable(tags)

        correct += torch.eq(predictions, targets).sum().data[0]

    return correct, len(data), correct/len(data)


def get_variable(x):
    """Get a Variable given indices x"""
    tensor = torch.cuda.LongTensor(x) if CUDA else torch.LongTensor(x)
    return Variable(tensor)


def preprocess(batch):
    """ Add zero-padding to a batch. """

    tags = [example.tag for example in batch]

    # add zero-padding to make all sequences equally long
    seqs = [example.words for example in batch]
    max_length = max(map(len, seqs))
    seqs = [seq + [PAD] * (max_length - len(seq)) for seq in seqs]

    return seqs, tags


optimizer = optim.Adam(model.parameters(), parameters.lr)

for ITER in range(parameters.epochs):

    random.shuffle(train)
    train_loss = 0.0
    start = time.time()
    updates = 0

    for batch in minibatch(train, parameters.batch_size):

        updates += 1

        # pad data with zeros
        seqs, tags = preprocess(batch)

        # forward pass
        scores = model(get_variable(seqs))
        targets = get_variable(tags)
        loss = nn.CrossEntropyLoss()
        output = loss(scores, targets)
        train_loss += output.data[0]

        # backward pass
        model.zero_grad()
        output.backward()

        # update weights
        optimizer.step()

    print("iter %r: avg train loss=%.4f, time=%.2fs" %
          (ITER, train_loss/updates, time.time()-start))

    # evaluate
    _, _, acc_train = evaluate(model, train)
    _, _, acc_dev = evaluate(model, dev)
    print("iter %r: train acc=%.4f  test acc=%.4f" % (ITER, acc_train, acc_dev))
