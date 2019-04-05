#!/usr/local/anaconda3/envs/experiments/bin/python3

import torch
import sys
from termcolor import colored
import random
from nltk.tokenize import word_tokenize

class lstm(torch.nn.Module):
    def __init__(self, hidden_size):
        super(lstm, self).__init__()

        self.embed = torch.nn.Embedding(alphabet_size, embed_size)
        self.forget = torch.nn.Linear(embed_size*n_chars+hidden_size, hidden_size)
        self.input = torch.nn.Linear(embed_size*n_chars+hidden_size, hidden_size)
        self.state = torch.nn.Linear(embed_size*n_chars+hidden_size, hidden_size)
        self.output = torch.nn.Linear(embed_size*n_chars+hidden_size, hidden_size)
        self.norm = torch.nn.BatchNorm1d(10)
        self.reset_hidden()

    def reset_hidden(self):
        self.h = torch.zeros(hidden_size).cuda()
        self.c = torch.zeros(hidden_size).cuda()

    def forward(self, input_vec):

        a = []

        for i in input_vec:
            a.append(self.embed(i))

        input_vec = torch.cat(a).cuda()

        h = self.h.detach()
        c = self.c.detach()

        input_vec = torch.cat((h, input_vec))

        f = self.forget(input_vec)
        i = self.input(input_vec)
        s = self.state(input_vec)
        o = self.output(input_vec)

        c = (c * f) + (i * s)
        h_ = torch.nn.Tanh()(c)
        h_ = o * h_

        self.c = c
        self.h = h_

        return h_

class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.encoder = lstm(hidden_size).cuda().float()
        self.decoder = lstm(hidden_size).cuda().float()

    def forward(self, seq):


filename = sys.argv[1]
text = []
alphabet = []

#open
with open(filename, "r") as f:
    # reads all lines and removes non alphabet words
    text = f.read()

f.close()

text = list(text)
tokens = set(text)

for i,e in enumerate(tokens):
    alphabet.append(e)

alphabet.sort()

print(alphabet)
epochs = 0

alphabet_size = len(alphabet)

#hyper parameters
hidden_size = 512
embed_size = 128
learning_rate = 0.00001

n_chars = 10
batch = 1
sequence_length = 5
step = 0
steps = 10000
c = 0
total_loss = 0.0
n_correct = 0
temperature = 0.5

render = False
greedy = True

one_hot_vecs = {}

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

def one_hot(char):

    t = torch.zeros(alphabet_size).cuda()
    if not char == " ":
        t[alphabet.index(char.lower())] = 1.0
    return t

def get_next_seq(c):
    i = 0

    char = alphabet.index(text[(c+i)%len(text)].lower())

    target = one_hot(text[(c+i+1)%len(text)])
    i += 1

    return char, target, c+i

counter = 0

#weights_init(encoder)
#weights_init(decoder)

counter = batch
loss_function = torch.nn.BCELoss()
optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())

step = int(random.choice(range(len(text))))
start = step
out_text = []
first = True

loss = torch.tensor(0.0, requires_grad=True).cuda()
letter = 0
c = 0
a = 0
head = [torch.tensor(0).cuda() for _ in range(n_chars)]

