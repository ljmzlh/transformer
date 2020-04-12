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

from model import transformer_model
from tensorboardX import SummaryWriter
import numpy as np
from Translator import Translator

import nltk

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


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


def cal_bleu(trg_seq,gen_seq,dl):
    trg_seq=trg_seq.transpose(0,1).tolist()
    ##gen_seq=gen_seq.transpose(0,1).cpu().numpy()
    gen_seq=gen_seq.view(1,-1).tolist()
    
    reference_allvocabs_key = "ref_allvocabs"
    gen_key = "gen"
    metric = cotk.metric.BleuCorpusMetric(dl,reference_allvocabs_key=reference_allvocabs_key,gen_key=gen_key)
    data = {reference_allvocabs_key: trg_seq,gen_key: gen_seq}
    metric.forward(data)
    bleu=metric.close()
    return bleu['bleu']


def patch_src(src, pad_idx):
    return src

def patch_trg(trg, pad_idx):
    trg, gold ,tt= trg[:-1, :], trg[1:, :].contiguous().view(-1),trg[:, :]
    return trg, gold,tt

def beam(model,src_seq,device,K=5,alpha=0.7):
    src_seq=src_seq.repeat(1,K)

    with torch.no_grad():
        trg_seq=torch.zeros((55,K),dtype=int).to(device)
        trg_seq[0][0]=2
        p=trg_seq
        p_prob=[0 for x in range(K)]
        gen=None
        gen_prob=-1e9
        n=1

        for now in range(50):
            q=torch.zeros(55,K*K)
            q_prob=[]
            m=0
            
            trg_seq=torch.tensor(p)
            _,prob=model(src_seq[:,:n],trg_seq[:now+1,:n],device)
            prob_seq=torch.nn.functional.softmax(prob.double(),dim=2,dtype=torch.double)
                
            
            if(now<49):
                t=torch.topk(prob_seq[now],K)

                index=t.indices.view(-1)
                    
                value=t.values.view(-1)

                for i in range(n):
                    for k in range(K):
                        x=index[k]
                        y=value[k]
                        trg_seq_tmp=trg_seq[:,i].clone()
                        
                        trg_seq_tmp[now+1]=x
                        ##prob_tmp=((p_prob[i]*now+np.log(y.cpu()))/(now+1)).clone()
                        prob_tmp=(p_prob[i]+np.log(y.cpu())).clone()
                        if(x!=3):
                            for j in range(now+2):
                                q[j,m]=trg_seq_tmp[j]
                            m+=1
                            q_prob.append(prob_tmp.clone())
                        else:
                            score=prob_tmp/((now+2) ** alpha)
                            if(gen_prob<score):
                                gen=trg_seq_tmp[:now+2].clone()
                                gen_prob=score
            else:
                x=3
                y=prob_seq[now][0][3]
                trg_seq_tmp=trg_seq[:,i].clone()
                trg_seq_tmp[now+1]=x
                prob_tmp=(p_prob[i]+np.log(y.cpu())).clone()
                score=prob_tmp/((now+2) ** alpha)
                if(gen_prob<score):
                    gen=trg_seq_tmp[:now+2].clone()
                    gen_prob=score
                    
            p=torch.zeros((55,K),dtype=int).to(device)
            p_prob=[]
            if(m==0):
                break
            n=K
            if(m<n):
                n=m 
            
            t=torch.topk(torch.tensor(q_prob),n)
            index=t.indices

            for i in range(n):
                x=index[i]
                for j in range(now+2):
                    p[j,i]=q[j,x]
                p_prob.append(q_prob[x].clone())
        
            if(n==0):
                break
            

    return gen





def test(translator,model,dm,device,opt,dl):

    model.eval()
    total_ppl,tot=0,0
    beam_tot=0

    debug=open('./debug','w')
    file=open('./result','w')
    with torch.no_grad():
        now=0
        key='dev'
        dm.restart(key, 1, shuffle=False)
        while True:
            incoming = get_next_batch(dm, key, restart=False)
            if incoming is None:
                break
            if(now>=0):
                src_seq = patch_src(incoming.data.post, opt.src_pad_idx).to(device)
            
                trg_seq,_,trg= map(lambda x: x.to(device), patch_trg(incoming.data.resp, opt.trg_pad_idx))

                _,prob = model(src_seq, trg_seq,device)
                
                t=src_seq
                ##pred_seq = translator.translate_sentence(t,device)

                ppl=cal_ppl(trg,prob,dl)['perplexity']
                    
                ##beam_gen=pred_seq
                beam_gen=beam(model,src_seq,device,K=opt.topk)
            
                beam_bleu=cal_bleu(trg,beam_gen,dl)
                print(beam_bleu)

                total_ppl+=np.log(ppl)

                beam_tot+=beam_bleu
                
                debug.write(str(now)+'\n')
                debug.write(str(src_seq.transpose(0,1))+'\n')
                debug.write(str(trg_seq.transpose(0,1))+'\n')
                debug.write(str(beam_gen)+'\n')
                debug.write(str(beam_bleu)+'\n')

                tot+=1
                print('dev ',now,' : ',ppl,beam_bleu)

                file.write('ppl: '+str(ppl)+' beam: '+str(beam_bleu)+'\n')

                   
            now=now+1

        ppl=np.exp(total_ppl/tot)
        beam_bleu=beam_tot/tot
        print(beam_bleu)
        file.write('ppl: '+str(ppl)+' beam: '+str(beam_bleu))
        file.write(str(tot)+'\n')

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




    translator = Translator(
        model=model,
        beam_size=opt.topk,
        max_seq_len=50,
        src_pad_idx=0,
        trg_pad_idx=0,
        trg_bos_idx=2,
        trg_eos_idx=3).to(device)

    dl=cotk.dataloader.OpenSubtitles(opt.datapath,min_vocab_times=data_arg.min_vocab_times)
    test(translator,model,dm,device,opt,dl)



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
