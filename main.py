#!/usr/local/anaconda3/envs/experiments/bin/python3

import torch
import sys
import datetime
from termcolor import colored
import atexit
import keyboard

#model
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

        self.decoder = torch.nn.Linear(hidden_size, alphabet_size)

        self.loss = torch.nn.BCELoss()
        print("Learning rate: ", rate)
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

        h_ = torch.nn.Dropout(0.0)(h_)
        d = self.decoder(h_)
        d = torch.nn.Softmax(dim=-1)(d/temperature)

        return h_, d
class gru(torch.nn.Module):
    def __init__(self, hidden_size, rate):
        super(lstm, self).__init__()

        self.forget_i = torch.nn.Linear(alphabet_size*sequence_length, hidden_size)
        self.forget_h = torch.nn.Linear(hidden_size, hidden_size)

        self.input_i = torch.nn.Linear(alphabet_size*sequence_length, hidden_size)
        self.input_h = torch.nn.Linear(hidden_size, hidden_size)

        self.output_i = torch.nn.Linear(alphabet_size*sequence_length, hidden_size)
        self.output_h = torch.nn.Linear(hidden_size, hidden_size)

        self.reset_hidden()

        self.decoder = torch.nn.Linear(hidden_size, alphabet_size)

        self.loss = torch.nn.BCELoss()
        print("Learning rate: ", rate)
        self.optim = torch.optim.Adam(lr=float(rate), params=self.parameters())

    def reset_hidden(self):
        self.c = torch.zeros(hidden_size).cuda()

    def forward(self, input_vec):

        input_vec = input_vec.detach()

        context = self.c.detach()

        a = torch.nn.Sigmoid()(self.forget_i(input_vec) + self.forget_h(c))
        b = torch.nn.Sigmoid()(self.input_i(input_vec) + self.input_h(c))

        o_h = context * a
        o_i = input_vec * a

        context = context * 1 - i


        o = torch.nn.Tanh()(self.output_i(o_i) + self.output_h(o_h))


        self.c = context

        h_ = torch.nn.Dropout(0.0)(c)
        d = self.decoder(h_)
        d = torch.nn.Softmax(dim=-1)(d/temperature)

        return h_, d
#load dataset
filename = sys.argv[1]
text = []
alphabet = []

#open file
with open(filename, "r") as f:
    # reads all lines and removes non alphabet words
    book = f.read()

f.close()

book = list(book)
text = []

#parse book
for t in book:
    if t == "\n":
        t = "¶"
    #print(t,end="")
    text.append(t)

for i,e in enumerate(book):
    if e.lower() not in alphabet:
        alphabet.append(e.lower())

del book

#sort/format tokens
alphabet.sort()
alphabet[alphabet.index("\n")] = "¶"
alphabet_size = len(alphabet)

epochs = 0

#parameters
hidden_size = 512
n_chars = 1
batch = 1
sequence_length = 6
step = 0
steps = 10000
rate = 0.00000065
c = 0
total_loss = 0.0
n_correct = 0
temperature = 1.0

render = False
show_grad = False

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
    t[alphabet.index(char.lower())] = 1.0
    return t

def get_next_seq(character):
    c = character
    new = False

    if c >= len(text):
        c = 0
        new = True

    i = 0
    out = []

    for i in range(sequence_length):
        char = one_hot(text[(c+i)%len(text)].lower())
        out.append(char)

    i+=1
    target = one_hot(text[(c+i)%len(text)].lower()).cuda()

    return out, target, c+1, new

def savemodel():
    print("Save model parameters? [y/n]➡")
    filename_input = input()

    if filename_input == 'y' or filename_input == 'Y' or filename_input.lower() == 'yes':
        filename = "Model-" + str(datetime.datetime.now()).replace(" ", "_")
        print("Save as filename [default: {}]➡".format(filename))

        filename_input = input()
        if not filename_input == "":
            filename = "Model-" + str(filename_input).replace(" ", "_")

        print("Saving model as {}...".format(filename))
        modelname = "./models/{}".format(filename)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optim.state_dict()
        }, modelname)

    # print("Best parameters:")
    # print(best)
    quit()

atexit.register(savemodel)

#init model
model = lstm(hidden_size, rate).cuda()
weights_init(model)

counter = 1

start = step
out_text = []

c = 1

for a in alphabet:
    print(a,end=" ")

print(" - Size:{}\n".format(alphabet_size))

#train loop
while True:

    inp = torch.zeros(alphabet_size)

    inp, target, c, new = get_next_seq(c)

    a = 0

    if new:
        if epochs % 100 == 0:
            print("")
            for i in range(int(len(out_text)/4)):
                print(out_text[i],end="")
            print("")

        out_text.clear()
        avg_loss = round(total_loss / len(text), 4)
        n_correct = 0
        counter = 0
        model.reset_hidden()
        epochs += 1

    d, out = model.forward(torch.cat(inp))

    probs = torch.distributions.Categorical(out)
    a = probs.sample()

    outchar = alphabet[a]
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
    #torch.nn.utils.clip_grad_norm_(model.parameters(), -1.0, 1.0)

    if keyboard.is_pressed(' '):
        show_grad = True

    if show_grad:
        for p in model.parameters():
            print(p)
            print(p.grad)
            break
        show_grad = False

    model.optim.step()
    model.optim.zero_grad()

    counter += 1

    accuracy = int(100*(n_correct/counter))
    p_avg = avg_loss
    if avg_loss > p_avg:
        indicator = "⬆"
    else:
        indicator = "⬇"

    out_text.append(outchar)

    progress = round(100*(c/len(text)),4)

    if targetchar == "¶":
        sys.stdout.flush()
        model.reset_hidden()

    if not render:
        if counter % 100 == 0:
            out = colored("\r[Epoch{}|Progress:[{}%]|[{}]|loss:{}|avg:{}|{}|Acc:{}%]".format(epochs, progress, a, 0, avg_loss, indicator, accuracy),attrs=['reverse'])
            print(out,end="")
    else:
        print(outchar,end="")


    del out, inp, target, outchar, targetchar
