from skip_thoughts.model import UniSkip, Encoder
from skip_thoughts.data_loader import DataLoader
from skip_thoughts.vocab import load_dictionary
from config import *
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch


class UsableEncoder:

    def __init__(self, loc=loc):
        print("Preparing the DataLoader. Loading the word dictionary")
        self.d = DataLoader(sentences=[''], word_dict=load_dictionary(data_skip))
        self.encoder = None

        print("Loading encoder from the saved model at {}".format(loc))
        model = UniSkip()
        model.load_state_dict(torch.load(loc, map_location=lambda storage, loc: storage))
        self.encoder = model.encoder
        if USE_CUDA:
            self.encoder.cuda(CUDA_DEVICE)

    def encode(self, text):
        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]

        ret = []

        for chunk in chunks(text, 3):
            print("encoding chunk of size {}".format(len(chunk)))
            indices = [self.d.convert_sentence_to_indices(sentence) for sentence in chunk]
            indices = torch.stack(indices)
            indices, _ = self.encoder(indices)
            indices = indices.view(-1, self.encoder.thought_size)
            indices = indices.data.cpu().numpy()

            ret.extend(indices)
        ret = np.array(ret)

        return ret


