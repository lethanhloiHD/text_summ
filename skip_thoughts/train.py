import torch
from torch import nn
from torch.autograd import Variable

from skip_thoughts.data_loader import DataLoader
from skip_thoughts.model import UniSkip
from config import *
from datetime import datetime, timedelta


data = DataLoader("data/data_skipthought/dummy_corpus.txt")
mod = UniSkip()
lr = 3e-4
optimizer = torch.optim.Adam(params=mod.parameters(), lr=lr)

loss_trail = []
last_best_loss = None
current_time = datetime.utcnow()


def debug(i, loss, prev, nex, prev_pred, next_pred):
    global loss_trail
    global last_best_loss
    global current_time

    this_loss = loss.item()
    loss_trail.append(this_loss)
    loss_trail = loss_trail[-20:]
    new_current_time = datetime.utcnow()
    time_elapsed = str(new_current_time - current_time)
    current_time = new_current_time
    print("Iteration {}: time = {} last_best_loss = {}, this_loss = {}".format(
        i, time_elapsed, last_best_loss, this_loss))

    print("prev = {}\nnext = {}\npred_prev = {}\npred_next = {}".format(
        data.convert_indices_to_sentences(prev),
        data.convert_indices_to_sentences(nex),
        data.convert_indices_to_sentences(prev_pred),
        data.convert_indices_to_sentences(next_pred),
    ))

    try:
        trail_loss = sum(loss_trail) / len(loss_trail)
        if last_best_loss is None or last_best_loss > trail_loss:
            print("Loss improved from {} to {}".format(last_best_loss, trail_loss))
            save_loc = "models/saved_models/skip-best".format(lr, VOCAB_SIZE)
            print("saving model at {}".format(save_loc))
            torch.save(mod.state_dict(), save_loc)

            last_best_loss = trail_loss
    except Exception as e:
        print("Couldn't save model because {}".format(e))


def train_skipthought():
    print("Starting training...")
    # a million iterations
    for i in range(0,10000):
        sentences, lengths = data.fetch_batch(32)
        loss, prev, nex, prev_pred, next_pred = mod(sentences, lengths)
        debug(i, loss, prev, nex, prev_pred, next_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()