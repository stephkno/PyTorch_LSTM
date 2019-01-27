#!/usr/local/anaconda3/envs/experiments/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import sys
import numpy
import datetime
import atexit
import math
from termcolor import colored
import tensorboardX

writer = tensorboardX.SummaryWriter()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if use_cuda:
    print("Using CUDA")
else:
    print("Using CPU")

results = dict()
examined = dict()

#parameter search mode - default off
search = False

#freerun mode - generate text from saved model
freerun = False

#generate parameter search table for parameter search mode
if search:
    # create parameter search space
    rate_parameters = []
    a = 0.05
    for r in range(100):
        a += 0.0005
        rate_parameters.append(round(a, 3))

    dropout_parameters = []
    for a in range(100):
        a = 0.0
        for r in range(100):
            a += 0.01
            dropout_parameters.append(round(a, 2))

    temperature_parameters = []
    for a in range(100):
        a = 0.1
        for r in range(100):
            a += 0.01
            temperature_parameters.append(round(a, 2))

    print(len(dropout_parameters))
    print(len(rate_parameters))

#defines single LSTM layer
class LSTM(nn.Module):

    def __init__(self, size, hidden, batch, prev, rate):
        super(LSTM, self).__init__()

        self.size = size
        self.rate = rate
        self.batch = 0
        self.epochs = 0
        self.prev = prev

        self.hidden_size = hidden * batch

        self.forget_input = torch.nn.Linear(size * prev * batch, self.hidden_size)
        self.forget_hidden = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.learn_input = torch.nn.Linear(size * prev * batch, self.hidden_size)
        self.learn_hidden = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.focus_input = torch.nn.Linear(size * prev * batch, self.hidden_size)
        self.focus_hidden = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.output_input = torch.nn.Linear(size * prev * batch, self.hidden_size)
        self.output_hidden = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.tanh = torch.nn.Tanh()

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=0)

        self.steps = 0
        self.generations = 0
        self.hidden = torch.zeros(self.hidden_size, device=device)

    def reset(self):
        #reset working memory and leave long term memory
        self.context = torch.zeros(self.hidden_size, device=device)

    def forward(self, x):
        # detach state tensors
        self.context = self.context.detach()
        self.hidden = self.hidden.detach()

        # layers
        x = x.detach().view(-1).cuda()

        # process layers
        f = torch.add(self.forget_input(x), self.forget_hidden(self.hidden))
        i = torch.add(self.learn_input(x), self.learn_hidden(self.hidden))
        s = torch.add(self.focus_input(x), self.focus_hidden(self.hidden))
        o = torch.add(self.output_input(x), self.output_hidden(self.hidden))

        # activations
        f = self.sigmoid(f)
        i = self.sigmoid(i)
        s = self.tanh(s)
        o = self.sigmoid(o)

        # gating mechanism
        self.context = (f * self.context) + (i * s)

        # tanh output
        self.hidden = self.tanh(self.context) * o

        return self.hidden.clone()
class Model(nn.Module):

    def __init__(self, size, prev, batch_size, dropout, rate, hidden):
        super(Model, self).__init__()

        #define two LSTM layers
        self.rnn1 = LSTM(size, hidden, batch_size, prev, rate).cuda()
        self.rnn2 = LSTM(hidden, hidden, batch_size, 1, rate).cuda()

        #linear decoder layer
        self.decoder = torch.nn.Linear(hidden * batch_size, size * batch_size).cuda()

        self.d = dropout
        self.r = rate

        self.epochs = 1
        self.batches = 1
        self.counter = 0
        self.runs = 0
        self.count = 0

        self.rate = rate

        self.dropout = torch.nn.Dropout(dropout)

        #field contains the current state of inputs for the model eg ["T", "H", "E".. etc
        self.field = [text[x] for x in range(n_prev)]

        #zeros out LSTM hidden and context vector
        self.clear_internal_states()

        #defines loss and optimizer
        self.loss_function = torch.nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam([
        #    {'params': self.parameters()},
        #    ], weight_decay=0.0, lr=rate)
        self.optimizer = torch.optim.SGD(params=self.parameters(), lr=rate, momentum=momentum)
        self.initialize_weights()

        #standard deviation for module parameters
        # std = 1.0/math.sqrt(self.rnn.hidden_size)

        # for p in self.parameters():
        #    p.data.uniform_(-std, std)

    def clear_internal_states(self):
        self.rnn1.reset()
        self.rnn2.reset()

    def initialize_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-1, 1)

    #converts array of chars into matrix for input
    def get_input_vector(self, chars):
        out = []

        for c in chars:
            out.append(one_hot(c))

        out = torch.stack(out).cuda()
        return out

    def forward(self, inp):
        x = torch.autograd.Variable((inp).view(-1))

        x = self.rnn1(x)
        x = self.dropout(x)

        x = self.rnn2(x)
        x = self.dropout(x)

        x = self.decoder(x)
        x = x.view(nbatches, -1)

        return x

def splash(a):
    if a:
        print(
            "RNN Text Generator\nUsage:\n\n-f --filename: filename of input text - required\n-h --hidden: number of hidden layers, default 1\n-r --rate: learning rate\n-p --prev: number of previous states to observe, default 0.05\n-t --temperature: sampling temperature")
        print("\nExample usage: <command> -f input.txt -h 128 -r 0.01234 -t 0.77")
    else:
        print("\nRNN Text Generator\n")
        print("Alphabet size: {}".format(alphabet_size))

        print("Hyperparameters:")
        params = sys.argv
        params.pop(0)

        for a in list(params):
            print(a, " ", end="")

        print("\n")
        print(datetime.datetime.now())
        print("\n")

def getIndexFromLetter(letter, list):
    return list.index(letter)

def getLetterFromIndex(i, list):
    return list[i]

#parse argument
def parse(args, arg):
    for i in range(len(args)):
        if args[i] in arg:
            if len(args) < i + 1:
                return ""
            if args[i + 1].startswith("-"):
                splash(True)
            else:
                return args[i + 1]

    return False

#save model parameters to a file
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
        modelname = "./gitignore/models/{}".format(filename)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'hidden_state_1': model.rnn1.hidden,
            'hidden_state_2': model.rnn2.hidden
        }, modelname)

    # print("Best parameters:")
    # print(best)
    quit()

#load model parameters from a file
def loadmodel():
    print("Load")
    # load model parameters if checkpoint specified
    if not model_filename == False:
        try:
            checkpoint = torch.load(model_filename)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.rnn1.hidden = checkpoint['hidden_state_1']
            model.rnn2.hidden = checkpoint['hidden_state_2']

        except FileNotFoundError:
            print("Model not found.")
            quit()
    else:
        print("New model")

#register exit event to save model
atexit.register(savemodel)
model_filename = None

#input arguments
try:
    model_filename = parse(sys.argv, ["--load", "-l"])
    filename = parse(sys.argv, ["--filename", "-f"])
    if not filename or filename == "":
        splash()
    rate = float(parse(sys.argv, ["--rate", "-r"]))
    if not rate or rate == "":
        rate = 0.0123456789
    hidden = int(parse(sys.argv, ["--hidden", "-h"]))
    if not hidden or hidden == "":
        hidden = 512
    nbatches = int(parse(sys.argv, ["--batch", "-b"]))
    if not nbatches:
        nbatches = 2
    momentum = float(parse(sys.argv, ["--momentum", "-m"]))
    if not momentum:
        momentum = 0.1
    n_prev = int(parse(sys.argv, ["--previous", "-p"]))
    if not n_prev:
        n_prev = 9
    dropout = float(parse(sys.argv, ["--dropout", "-d"]))
    if not dropout:
        dropout = 0.5
    temperature = float(parse(sys.argv, ["--temp", "-t"]))
    if not temperature:
        temperature = 0.77

except:
    splash(True)
    quit()

#initial symbol space
alphabet = [' ', '!', '"', '#', '$', '%', '&', "'",
            '(', ')', '*', '+', ',', '-', '.', '/',
            '0', '1', '2', '3', '4', '5', '6', '7',
            '8', '9', ':', ';', '<', '=', '>', '?',
            '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
            'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
            'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
            'X', 'Y', 'Z', '[', ']', '^', '_', 'a',
            'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
            'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
            'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
            'z', '|', '~', '¶']

#defines array for subject text
text = []
e = 0
c = 0

# open file
with open(filename, "r") as f:
    # reads all lines and removes non alphabet words
    intext = f.read()

for l in list(intext):
    if l == "\n": l = "¶"
    if l == "\x1b": print("XXX")
    text.append(l)

for l in text:
    sys.stdout.flush()

    if l not in alphabet:
        alphabet.append(l)
        print("\r{}% - {}/{}".format(int(c / len(text) * 100), c, len(text)), end="")
    c += 1

#very important for symbol space to be in alphabetical order
alphabet.sort()
alphabet_size = len(alphabet)

#load model
if not model_filename == None:
    loadmodel()

# initialize and/or reset model for parameter search mode
def reset_model():

    # main cycle of training program begins
    while True:
        # splash(False)

        r = random.choice(rate_parameters)
        rate_parameters.remove(r)
        #d = random.choice(dropout_parameters)
        #dropout_parameters.remove(d)
        #t = random.choice(temperature_parameters)
        #temperature_parameters.remove(t)
        d = 0.5

        model = Model(alphabet_size, n_prev, nbatches, d, r, hidden).cuda()
        model.counter = n_prev - 1

        print("\nParameters: \n -Rate: {} \n -Dropout: {}\n -Temperature: {}".format(r, d, temperature))

        train_cycle(model, temperature)

# encode vector from char
def one_hot(char):
    output = torch.zeros(alphabet_size).cuda()
    output[alphabet.index(char)] = 1

    return output

# get output char from vectors
def get_output(inp, t):
    inp = torch.nn.Softmax(dim=0)((inp / t).exp())
    #sample = torch.multinomial(inp / inp.sum(), 1)[:]
    inp = inp / inp.sum()
    inp = inp.cpu().detach().numpy()

    sample = numpy.random.choice(alphabet, p=inp)
    return sample

# training cycle -- runs until net gets stuck in loop
def train_cycle(model, temperature):
    while True:

        t = 0

        total_time = 0
        total_loss = 0

        #1000 runs per minibatch
        steps = 350
        model.runs += 1
        print("")

        if not freerun:
            while t < steps:

                # get target char
                new_letter = alphabet.index(text[(model.counter + 1) % len(text)])

                # make target vector
                target = [new_letter for _ in range(nbatches)]
                target = torch.tensor(target).cuda()

                inp = model.get_input_vector(model.field)
                inp = [inp for _ in range(nbatches)]
                inp = torch.stack(inp)

                inp = torch.nn.functional.normalize(inp, dim=0)
                out = model.forward(inp)

                # get outputs
                char = []

                for o in out.split(1):
                    a = get_output(o[0], temperature)

                    # a = alphabet[torch.argmax(out)]
                    char.append(a)

                # get input text
                f = ""
                for z in model.field:
                    f += z
                # get output text
                c = ''.join(str(e) for e in char)

                progress = int(100 * (t / steps))

                #display some stats and progress
                if t % 10 == 0:
                    txt = colored(
                        "\r ▲ {} | Training | Progress: {}% | {}/{} | Epoch: {} | Batch: {} | {} | {} | {} |".format(
                            model.count, progress, model.counter, len(text), model.epochs, model.batches, f, c,
                            alphabet[new_letter]),
                        attrs=['reverse'])
                    print(txt, end="")

                #print each character in a series
                # print(c, end="")
                # sys.stdout.flush()

                #increment counters
                model.counter += 1
                t += 1

                #calculate loss for pass
                loss = model.loss_function(out, target)
                total_loss += loss.item()
                model.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                model.optimizer.step()

                #check if we have reached end of text file
                if model.counter > len(text):
                    model.epochs += 1
                    model.batches = 0
                    model.counter = 0
                    print("\nNew Epoch")

                #appends new letter to end of field array
                model.field.append(alphabet[new_letter])
                model.field.pop(0)

            #tensorboardx
            writer.add_scalar('time', torch.tensor(total_time), model.runs)

            #minibatch end stats
            print("\nAvg Loss: {} | Generating text...".format(total_loss / t))
            model.count += 1
            sys.stdout.flush()
            del total_loss
            torch.cuda.empty_cache()

            model.batches += 1
            total_loss = 0
            t = 0

            writer.add_scalar('total_loss', int(total_loss), model.counter)

        variety = []
        if model.runs % 10 == 0:
            steps = 3000
        else:
            steps = 300

        #free generation cycle
        for i in range(steps):
            if freerun: steps = 0
            # print(model.field,end="")
            # print(" ",end="")
            inp = model.get_input_vector(model.field)
            inp = [inp for _ in range(nbatches)]
            inp = torch.stack(inp)
            out = model.forward(inp)

            # get outputs
            char = []
            output = []

            for o in out.split(1):
                a = get_output(o[0], temperature)
                char.append(a)

            c = char[0]

            if c not in variety:
                variety.append(c)

            if model.runs % 1 == 0:
                print(c, end="")
                sys.stdout.flush()

            model.field.append(c)
            model.field.pop(0)

        model.clear_internal_states()

        variety = int(100 * (len(variety) / alphabet_size))
        print("\nVariety: {}\n".format(variety))
        writer.add_scalar('variety', variety, model.runs)
        # for p in model.parameters():
        #    print(p)
        #if parameter search mode enabled test if network is failing or succeeding/stuck
        if search:
            if variety < 5:
                archive = str("d:{}r:{}t:{}".format(model.d, model.r, temperature))
                return
            if model.count > 999:
                archive = str("dropout: {} rate: {} temperature: {}".format(model.d, model.r, temperature))
                return

# initialize program
if search:
    #initialize parameter search
    reset_model()
else:
    #initialize training cycle
    model = Model(alphabet_size, n_prev, nbatches, dropout, rate, hidden).cuda()
    train_cycle(model, temperature)
