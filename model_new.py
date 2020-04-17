import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import EncoderDecoder,Encoder,Decoder,Generator
from sublayer import EncoderLayer,DecoderLayer
from module import MultiHeadedAttention,PositionwiseFeedForward,PositionalEncoding,subsequent_mask

from torch.autograd import Variable
import numpy as np

##Transformer model is borrowed from Harvard NLP: The Annotated Transformer

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

def make_model(opt,src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),opt)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

def transformer_model(opt,device):
    model = make_model(opt,
        src_vocab=opt.n_src_vocab,tgt_vocab=opt.n_trg_vocab, N=opt.n_layer,
        d_model=opt.d_model,d_ff=opt.d_inner,h=opt.n_head,dropout=opt.dropout)
    return model