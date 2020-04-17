import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc

from Optim import ScheduledOptim

from utils import debug, try_cache, cuda_init, Storage,cuda
from cotk.dataloader import SingleTurnDialog
from cotk.wordvector import WordVector, Glove
import cotk

from model_new import transformer_model
from tensorboardX import SummaryWriter
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def cal_loss(pred, gold, trg_pad_idx):

    loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx,reduction='sum')
    return loss


def cal_performance(pred, gold, trg_pad_idx):

    loss = cal_loss(pred, gold, trg_pad_idx)
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_word = non_pad_mask.sum().item()
    
    return loss, n_word

def cal_ppl(trg_seq,prob,dl):
    trg_seq=trg_seq.transpose(0,1)
    prob=prob.transpose(0,1)
    
    
    metric= cotk.metric.PerplexityMetric(dl)
    prob_seq=torch.nn.functional.log_softmax(prob.double(),dim=2,dtype=torch.double)
    
    len_seq=[]
    for i in range(trg_seq.size(0)):
        t=trg_seq.size(1)
        for j in range(trg_seq.size(1)):
            if(trg_seq[i][j]==3):
                t=j+1
                break
        len_seq.append(t)
        
        
    reference_allvocabs_key="ref_allvocabs"
    reference_len_key="ref_length"
    gen_log_prob_key="gen_log_prob"
    data={reference_allvocabs_key: trg_seq, reference_len_key: len_seq, gen_log_prob_key:prob_seq}
    reference_allvocabs_key
    metric.forward(data)
    
    ppl=metric.close()
    return ppl

def patch_src(src, pad_idx):
    return src

def patch_trg(trg, pad_idx):
    trg, gold ,tt= trg[:-1, :], trg[1:, :].contiguous().view(-1),trg[:, :]
    return trg, gold,tt

def train_epoch(now_epoch,model,dm,optimizer,device,opt,writer,dl):
    
    global global_step

    model.train()
    total_loss, n_word_total= 0, 0 
    now=0
    key='train'
    dm.restart(key, opt.batch_size, shuffle=False)
    optimizer.zero_grad()
    while True:
        incoming = get_next_batch(dm, key, restart=False)
        if incoming is None:
            break

        src_seq = patch_src(incoming.data.post, opt.src_pad_idx).to(device)
        trg_seq, gold , _= map(lambda x: x.to(device), patch_trg(incoming.data.resp, opt.trg_pad_idx))

        pred,prob = model(src_seq, trg_seq)
        ##print(prob.shape)
        loss, n_word = cal_performance(pred, gold, opt.trg_pad_idx) 

        loss=loss/opt.grad_step

        loss.backward()

        if((now+1)%opt.grad_step==0):
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step_and_update_lr()
            global_step+=1
            optimizer.zero_grad()

        n_word_total+=n_word
        total_loss+=loss.item()*opt.grad_step

        if((now+1)%opt.grad_step==0):
            res_loss=total_loss/n_word_total
            print('epoch ',now_epoch,' train ',now,' : ',res_loss)
            writer.add_scalar('train_loss',res_loss,global_step)
            total_loss,n_word_total=0,0

        now=now+1
        if(global_step%opt.save_step==0):
            state={'net':model.state_dict(),'opt':optimizer._optimizer.state_dict(),'n_steps':optimizer.n_steps}
            torch.save(state,'./train_state')

    
def dev_epoch(now_epoch,model,dm,optimizer,device,opt,writer,dl):
    
    global global_min

    model.eval()
    total_ppl,total_batch= 0,0
    with torch.no_grad():
        now=0
        key='dev'
        dm.restart(key, opt.batch_size, shuffle=False)
        while True:
            incoming = get_next_batch(dm, key, restart=False)
            if incoming is None:
                break
            src_seq = patch_src(incoming.data.post, opt.src_pad_idx).to(device)
            trg_seq, gold , trg= map(lambda x: x.to(device), patch_trg(incoming.data.resp, opt.trg_pad_idx))

            _,prob = model(src_seq, trg_seq)
            
            ppl=cal_ppl(trg,prob,dl)['perplexity']
            
            ppl=np.log(ppl)

            total_ppl+=ppl
            
            now=now+1
            total_batch+=1
            print('epoch ',now_epoch,' dev ',now,' : ',np.exp(ppl))

    dev_ppl=np.exp(total_ppl/total_batch)
    writer.add_scalar('dev_ppl',dev_ppl,now_epoch)

    state={'net':model.state_dict(),'opt':optimizer._optimizer.state_dict(),'n_steps':optimizer.n_steps}
    torch.save(state,'./dev_last')

    if(global_min>dev_ppl):
        global_min=dev_ppl
        state={'net':model.state_dict(),'opt':optimizer._optimizer.state_dict(),'n_steps':optimizer.n_steps}
        torch.save(state,'./dev_best')

def train(model,dm,optimizer,device,opt,dl):

    writer= SummaryWriter()

    for i in range(opt.epoch):
        train_epoch(i,model,dm,optimizer,device,opt,writer,dl)
        dev_epoch(i,model,dm,optimizer,device,opt,writer,dl)
        
def train_process(opt):
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model
    opt.batch_size=opt.b

    device = torch.device('cuda' if opt.cuda else 'cpu')

    data_class = SingleTurnDialog.load_class('OpenSubtitles')
    data_arg = Storage()
    data_arg.file_id = opt.datapath
    data_arg.min_vocab_times=20

    def load_dataset(data_arg, wvpath, embedding_size):
        dm = data_class	(**data_arg)
        return dm

    opt.n_position=100
    dm= load_dataset(data_arg, None, opt.n_position)

    opt.n_src_vocab=dm.valid_vocab_len
    opt.n_trg_vocab=dm.valid_vocab_len
    opt.n_vocab_size=dm.valid_vocab_len
    opt.src_pad_idx=0
    opt.trg_pad_idx=0
    opt.pad_idx=0

    model=transformer_model(opt,device).to(device)

    
    n_steps=0
    optimizer_=optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09)

    if(opt.restore!=None):
        checkpoint=torch.load(opt.restore)
        model.load_state_dict(checkpoint['net'])
        n_steps=checkpoint['n_steps']
        optimizer_.load_state_dict(checkpoint['opt'])
    
    optimizer = ScheduledOptim(optimizer_,opt.lr, opt.d_model, opt.n_warmup_steps,n_steps)

    
    dl=cotk.dataloader.OpenSubtitles(opt.datapath,min_vocab_times=data_arg.min_vocab_times)
    train(model,dm,optimizer,device,opt,dl)





def main(opt):
    from test import test_process

    if(opt.mode=='train'):
        train_process(opt)
    elif(opt.mode=='test'):
        test_process(opt)
    else:
        raise ValueError("Unknown mode")

def _preprocess_batch(data):
		incoming = Storage()
		incoming.data = data = Storage(data)
		data.batch_size = data.post.shape[0]
		data.post = cuda(torch.LongTensor(data.post.transpose(1, 0))) 
		data.resp = cuda(torch.LongTensor(data.resp.transpose(1, 0))) 
		return incoming

def get_next_batch(dm, key, restart=True):
		data = dm.get_next_batch(key)
		if data is None:
			if restart:
				dm.restart(key)
				return get_next_batch(dm, key, False)
			else:
				return None
		return _preprocess_batch(data)


global_step=0
global_min=1e9

