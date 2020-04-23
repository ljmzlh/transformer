import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import debug, try_cache, cuda_init, Storage,cuda
from cotk.dataloader import SingleTurnDialog
from cotk.wordvector import WordVector, Glove
import cotk

from model_new import transformer_model
from tensorboardX import SummaryWriter
import numpy as np

import nltk

import os


def cal_loss(pred, gold, trg_pad_idx):

    loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx,reduction='sum')
    return loss


def cal_performance(pred, gold, trg_pad_idx):

    loss = cal_loss(pred, gold, trg_pad_idx)
    pred = pred.max(1)[1]
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
        len_seq.append(trg_seq.size(1))
        
    
    reference_allvocabs_key="ref_allvocabs"
    reference_len_key="ref_length"
    gen_log_prob_key="gen_log_prob"
    data={reference_allvocabs_key: trg_seq, reference_len_key: len_seq, gen_log_prob_key:prob_seq}
    

    metric.forward(data)
    
    ppl=metric.close()
    return ppl

def cal_bleu(src_seq,trg_seq,gen_seq,debug,dl):
    return 1


def cal_bleu1(trg_seq,gen_seq,dl):
    trg_seq=trg_seq.transpose(0,1).detach().cpu().numpy()
    gen_seq=gen_seq.transpose(0,1).cpu().numpy()
    
    reference_allvocabs_key = "ref_allvocabs"
    gen_key = "gen"
    metric = cotk.metric.BleuCorpusMetric(dl,reference_allvocabs_key=reference_allvocabs_key,gen_key=gen_key)
    data = {reference_allvocabs_key: trg_seq,gen_key: gen_seq}
    metric.forward(data)
    bleu=metric.close()

    print(bleu['bleu'])
    return bleu['bleu']





def patch_src(src, pad_idx):
    return src

def patch_trg(trg, pad_idx):
    trg, gold ,tt= trg[:-1, :], trg[1:, :].contiguous().view(-1),trg[:, :]
    return trg, gold,tt

def gen(model,src_seq,device):
    trg_seq=torch.zeros((55,1),dtype=int).to(device)
    trg_seq[0][0]=2
    for now in range(50):
        pred,prob=model(src_seq,trg_seq)
        pred=pred.max(1)[1]
        trg_seq[now+1][0]=pred[now]
        if(pred[now]==3)or(now==49):
            return trg_seq[:now+2]

def beam(model,src_seq,device,K=2):
    with torch.no_grad():
        trg_seq=torch.zeros((55,1),dtype=int).to(device)
        trg_seq[0][0]=2
        p=[trg_seq]
        p_prob=[0]
        gen=None
        gen_prob=-1e9

        for now in range(50):
            q,q_prob=[],[]
            for i in range(len(p)):
                trg_seq=p[i].clone()
                _,prob=model(src_seq,trg_seq[:now+1])
                prob_seq=torch.nn.functional.softmax(prob.double(),dim=2,dtype=torch.double)
                
            
                if(now<49):
                    t=torch.topk(prob_seq[now],K)
                    index=t.indices.view(-1)
                    
                    value=t.values.view(-1)
                    for k in range(K):
                        x=index[k]
                        y=value[k]
                        trg_seq_tmp=trg_seq.clone()
                        trg_seq_tmp[now+1][0]=x
                        ##prob_tmp=((p_prob[i]*now+np.log(y.cpu()))/(now+1)).clone()
                        prob_tmp=(p_prob[i]+np.log(y.cpu())).clone()
                        if(x!=3):
                            q.append(trg_seq_tmp.clone())
                            q_prob.append(prob_tmp.clone())
                        else:
                            if(gen_prob<prob_tmp):
                                gen=trg_seq_tmp[:now+2].clone()
                                gen_prob=prob_tmp.clone()
                else:
                    x=3
                    y=prob_seq[now][0][3]
                    trg_seq_tmp=trg_seq.clone()
                    trg_seq_tmp[now+1][0]=x
                    prob_tmp=(p_prob[i]+np.log(y.cpu())).clone()
                    if(gen_prob<prob_tmp):
                        gen=trg_seq_tmp[:now+2].clone()
                        gen_prob=prob_tmp.clone()
                    
            p=[]
            p_prob=[]
            if(len(q)==0):
                break
            l=K
            if(len(q)<l):
                l=len(q)
            index=torch.topk(torch.tensor(q_prob),l).indices.view(-1)
            for i in range(l):
                x=index[i]
                p.append(q[x].clone())
                p_prob.append(q_prob[x].clone())
        
            if(len(p)==0):
                break


        
    return gen




def test(model,dm,device,opt,dl):

    model.eval()
    total_ppl,tot=0,0
    valid_ppl=0
    greedy_valid,greedy_valid1,beam_valid,beam_valid1=0,0,0,0
    greedy_tot,greedy_tot1,beam_tot,beam_tot1=0,0,0,0

    debug=open('./debug','w')
    file=open('./result','w')
    with torch.no_grad():
        now=0
        key='test'
        dm.restart(key, 1, shuffle=False)
        while True:
            incoming = get_next_batch(dm, key, restart=False)
            if incoming is None:
                break
            if(now>=0):
                src_seq = patch_src(incoming.data.post, opt.src_pad_idx).to(device)
            
                trg_seq, gold , trg= map(lambda x: x.to(device), patch_trg(incoming.data.resp, opt.trg_pad_idx))

                pred,prob = model(src_seq, trg_seq)

                loss, n_word = cal_performance(pred, gold, opt.trg_pad_idx) 

                ppl=cal_ppl(trg,prob,dl)['perplexity']
                    
                import time

                start = time.time()
                greedy_gen=gen(model,src_seq,device)
                end = time.time()
                print("Execution Time: ", end - start)

                start = time.time()
                beam_gen=beam(model,src_seq,device,K=5)
                end = time.time()
                print("Execution Time: ", end - start)
               
                greedy_bleu=cal_bleu(src_seq,trg[1:,:],greedy_gen,debug,dl)
                greedy_bleu1=cal_bleu1(trg,greedy_gen,dl)
                beam_bleu=cal_bleu(src_seq,trg[1:,:],beam_gen,debug,dl)
                beam_bleu1=cal_bleu1(trg,beam_gen,dl)

                total_ppl+=np.log(ppl)
                valid_ppl+=1

                greedy_tot+=greedy_bleu
                greedy_valid+=1

                greedy_tot1+=greedy_bleu1
                greedy_valid1+=1
                
                beam_tot+=beam_bleu
                beam_valid+=1

                beam_tot1+=beam_bleu1
                beam_valid1+=1
                
                debug.write(str(now)+'\n')
                debug.write(str(src_seq.transpose(0,1))+'\n')
                debug.write(str(trg_seq.transpose(0,1))+'\n')
                debug.write(str(greedy_gen.transpose(0,1))+'\n')
                debug.write(str(beam_gen.transpose(0,1))+'\n')
                debug.write(str(greedy_bleu)+'\n')
                debug.write(str(greedy_bleu1)+'\n')
                debug.write(str(beam_bleu)+'\n')
                debug.write(str(beam_bleu1)+'\n')


                tot+=1
                print('dev ',now,' : ',ppl,greedy_bleu1,beam_bleu1)

                file.write('ppl: '+str(ppl)+' greedy: '+str(greedy_bleu1)+' beam: '+str(beam_bleu1))
                file.write('\n')
                   
            now=now+1

        ppl=np.exp(total_ppl/valid_ppl)
        greedy_bleu=greedy_tot/greedy_valid
        greedy_bleu1=greedy_tot1/greedy_valid1
        beam_bleu=beam_tot/beam_valid
        beam_bleu1=beam_tot1/beam_valid1
        print(ppl,greedy_bleu1,beam_bleu1)
        file.write('ppl: '+str(ppl)+' greedy: '+str(greedy_bleu)+' '+str(greedy_bleu1)+
                ' beam: '+str(beam_bleu)+' '+str(beam_bleu1))
        file.write(str(tot)+'\n')
        file.write(str(beam_valid)+'\n')
        file.write(str(beam_valid1)+'\n') 

        debug.close()
        file.close()




def test_process(opt):
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

    if(opt.restore!=None):
        checkpoint=torch.load(opt.restore)
        model.load_state_dict(checkpoint['net'])

    dl=cotk.dataloader.OpenSubtitles(opt.datapath,min_vocab_times=data_arg.min_vocab_times)
    test(model,dm,device,opt,dl)



def _preprocess_batch(data):
		incoming = Storage()
		incoming.data = data = Storage(data)
		data.batch_size = data.post.shape[0]
		data.post = cuda(torch.LongTensor(data.post.transpose(1, 0))) # length * batch_size
		data.resp = cuda(torch.LongTensor(data.resp.transpose(1, 0))) # length * batch_size
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
