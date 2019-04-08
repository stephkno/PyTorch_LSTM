#!/usr/local/anaconda3/envs/experiments/bin/python3

import torch
import sys
from termcolor import colored
import random
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

class lstm(torch.nn.Module):
    def __init__(self, hidden_size, rate):
        super(lstm, self).__init__()

        self.forget_i = torch.nn.Linear(alphabet_size*sequence_length, hidden_size)
        self.forget_h = torch.nn.Linear(hidden_size, hidden_size)

        self.input_i = torch.nn.Linear(alphabet_size*sequence_length, hidden_size)
        self.input_h = torch.nn.Linear(hidden_size, hidden_size)

        self.state_i = torch.nn.Linear(alphabet_size*sequence_length, hidden_size)
        self.state_h = torch.nn.Linear(hidden_size, hidden_size)

        self.output_i = torch.nn.Linear(alphabet_size*sequence_length, hidden_size)
        self.output_h = torch.nn.Linear(hidden_size, hidden_size)

        self.reset_hidden()
        self.decoder1 = torch.nn.Linear(hidden_size, int(hidden_size/2))
        self.decoder2 = torch.nn.Linear(int(hidden_size/2), int(hidden_size/3))
        self.decoder3 = torch.nn.Linear(int(hidden_size/3), alphabet_size)

        self.loss = torch.nn.BCELoss()
        print(rate)
        self.optim = torch.optim.Adam(lr=float(rate), params=self.parameters())

    def reset_hidden(self):
        self.h = torch.zeros(hidden_size).cuda()
        self.c = torch.zeros(hidden_size).cuda()

    def forward(self, input_vec):

        input_vec = input_vec.detach()

        h = self.h.detach()
        c = self.c.detach()

        f = torch.nn.Sigmoid()(self.forget_i(input_vec) + self.forget_h(h))
        i = torch.nn.Sigmoid()(self.input_i(input_vec) + self.input_h(h))
        s = torch.nn.Tanh()(self.state_i(input_vec) + self.state_h(h))
        o = torch.nn.Sigmoid()(self.output_i(input_vec) + self.output_h(h))

        c = (c * f) + (i * s)
        h_ = torch.nn.Tanh()(c)
        h_ = o * h_

        self.c = c
        self.h = h_

        d = self.decoder1(h_)
        d = torch.nn.Tanh()(d)
        d = self.decoder2(d)
        d = torch.nn.Tanh()(d)
        d = self.decoder3(d)
        d = torch.nn.Softmax(dim=-1)(d/temperature)

        return h_, d

filename = sys.argv[1]
text = []
alphabet = []

#open
with open(filename, "r") as f:
    # reads all lines and removes non alphabet words
    book = f.read()

f.close()

sent_token = book
text = []

for t in sent_token:
    text.append(t)

for i,e in enumerate(book):
    if e.lower() not in alphabet:
        alphabet.append(e.lower())

alphabet.sort()
alphabet_size = len(alphabet)

epochs = 0

#hyper parameters
hidden_size = 32
n_chars = 1
batch = 1
sequence_length = 2
step = 0
steps = 10000
rate = 0.0001
c = 0
total_loss = 0.0
n_correct = 0
temperature = 1.0

render = True
greedy = True

one_hot_vecs = {}
p_avg = 0
avg_loss = 0

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

def get_next_seq(character):
    c = character
    new = False
    seq = text

    if c >= len(text):
        sequence = 0
        c = 0
        sys.stdout.flush()
        new = True

    i = 0
    out = []

    for i in range(sequence_length):
        char = one_hot(seq[(c+i)%len(seq)].lower())
        out.append(char)

    i+=1
    target = one_hot(seq[(c+i)%len(seq)].lower()).cuda()

    return out, target, c+1, new

#embed = W2V(alphabet_size, embed_size).cuda()
model = lstm(hidden_size, rate).cuda()

#weights_init(embed)
weights_init(model)

counter = 1

start = step
out_text = []
first = True

letter = 0
c = 1
a = 0
sequence = 0

for a in alphabet:
    print(a,end=" ")

print(" - Size:{}\n".format(alphabet_size))

total_reset = -1

#outer loop runs forever
while True:
    total_reset += 1
    #inner loop breaks if nan output
    while True:

        inp = torch.zeros(alphabet_size)

        inp, target, c, new = get_next_seq(c)

        a = 0

        if new:
            for o in out_text:
                print(o,end="")
            out_text.clear()
            avg_loss = round(total_loss / len(text), 4)
            n_correct = 0
            counter = 0

        d, out = model.forward(torch.cat(inp))

        probs = torch.distributions.Categorical(out)
        a = probs.sample()

        outchar = alphabet[a]
        #outchar = alphabet[torch.argmax(out)]
        targetchar = alphabet[torch.argmax(target)]

        done = False

        if outchar == targetchar:
            a = "✅"
            n_correct += 1
            done = True
        else:
            a = "❌"

        loss = model.loss(out.view(-1), target.view(-1))
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), -1.0, 1.0)

    #    for p in model.parameters():
    #        print(p.grad)

        model.optim.step()
        model.optim.zero_grad()

        total_loss += loss.item()
        counter += 1

        accuracy = int(100*(n_correct/counter))
        p_avg = avg_loss
        if avg_loss > p_avg:
            indicator = "⬆"
        else:
            indicator = "⬇"

        out_text.append(outchar)

        progress = int(100*(sequence/len(text)))

        if not render:
            if outchar == "\n":
                outchar = "/n"
            if targetchar == "\n":
                targetchar = "/n"

            if counter % 100 == 0:
                print("\r[Search{}|Epoch{}|Progress{}%|{}|{}|{}|loss:{}|avg:{}|{}|Acc:{}%|lr:{}]".format(total_reset, epochs, progress, outchar, targetchar, a, loss.item()/counter, avg_loss, indicator, accuracy, rate),end="")
        else:
            if outchar == "	":
                outchar = ">"
            print(outchar,end="")
            #print(targetchar)

            sys.stdout.flush()

        del out, inp, target, outchar, targetchar
        torch.cuda.empty_cache()