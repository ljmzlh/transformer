import torch.nn as nn
import numpy as np
import torch

class PositionalEncoding(nn.Module):
    def __init__(self,d_hid,n_position=200):
        super(PositionalEncoding,self).__init__()
        self.register_buffer('pos_table',self._get_sinusoid_encoding_table(n_position,d_hid))
    
    def _get_sinusoid_encoding_table(self,n_position,d_hid):
        
        def get_position_angle_vec(position):
            return [position/np.power(10000,2*(hid_j//2)/d_hid) for hid_j in range(d_hid)]
        
        sinusoid_table=np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:,0::2]=np.sin(sinusoid_table[:,0::2])
        sinusoid_table[:,1::2]=np.cos(sinusoid_table[:,1::2])
        
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    
    def forward(self,x):
        return (x.transpose(0,1)+self.pos_table[:,:x.size(0)].clone().detach()).transpose(0,1)


def get_pad_mask(seq,pad_idx):
	return (seq==pad_idx).transpose(0,1)

class transformer_model(nn.Module):
    def __init__(self,opt,device):
        super().__init__()
        self.src_pad_idx=opt.src_pad_idx
        self.trg_pad_idx=opt.trg_pad_idx

        self.transformer=nn.Transformer(d_model=opt.d_model,nhead=opt.n_head,
        num_encoder_layers=opt.n_layer,num_decoder_layers=opt.n_layer,
        dim_feedforward=opt.d_inner,dropout=opt.dropout).to(device)

        self.scr_word_emb=nn.Embedding(opt.n_src_vocab,opt.d_word_vec,padding_idx=opt.pad_idx)
        self.src_pe=PositionalEncoding(opt.d_word_vec,n_position=opt.n_position)
        self.src_drop=nn.Dropout(p=opt.dropout)

        self.trg_word_emb=nn.Embedding(opt.n_trg_vocab,opt.d_word_vec,padding_idx=opt.pad_idx)
        self.trg_pe=PositionalEncoding(opt.d_word_vec,n_position=opt.n_position)
        self.trg_drop=nn.Dropout(p=opt.dropout)

        self.trg_word_prj=nn.Linear(opt.d_model,opt.n_trg_vocab)

    
    def forward(self,src,trg,device):

        src_mask=get_pad_mask(src,self.src_pad_idx).to(device)
        trg_mask=get_pad_mask(trg,self.trg_pad_idx).to(device)
        tgt_mask=self.transformer.generate_square_subsequent_mask(trg.size(0)).to(device)
        

        t0=self.scr_word_emb(src)
        t=self.src_pe(t0)
        src_seq=self.src_drop(t)
        trg_seq=self.trg_drop(self.trg_pe(self.trg_word_emb(trg)))

        ##print(tgt_mask)
        out=self.transformer(src_seq,trg_seq,src_key_padding_mask=src_mask,tgt_key_padding_mask=trg_mask,tgt_mask=tgt_mask)
        seq_logit=self.trg_word_prj(out)

        return seq_logit.view(-1,seq_logit.size(2)),seq_logit