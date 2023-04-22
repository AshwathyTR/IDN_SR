#!/usr/bin/env python
#coding:utf8
from .BasicModule import BasicModule
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .Attention import Attention
from torch.autograd import Variable

class wordonlyAttnRNNW(BasicModule):
    def __init__(self, args, embed=None):
        super(wordonlyAttnRNNW,self).__init__(args)
        self.model_name = 'wordonlyAttnRNNW'
        self.args = args
        drop = self.args.dropout
        V = args.embed_num
        D = args.embed_dim
        H = args.hidden_size
        S = args.seg_num

        P_V = args.pos_num
        P_D = args.pos_dim
        self.abs_pos_embed = nn.Embedding(P_V,P_D)
        self.rel_pos_embed = nn.Embedding(S,P_D)
        self.embed = nn.Embedding(V,D,padding_idx=0)
        if embed is not None:
            self.embed.weight.data.copy_(embed)

        self.attn = Attention()
        self.word_query = nn.Parameter(torch.randn(1,1,2*H))
        self.sent_query = nn.Parameter(torch.randn(1,1,2*H))

        self.word_RNN = nn.GRU(
                        input_size = D,
                        hidden_size = H,
                        batch_first = True,
                        bidirectional = True
                        )
        self.sent_RNN = nn.GRU(
                        input_size = 2*H,
                        hidden_size = H,
                        batch_first = True,
                        bidirectional = True
                        )
               
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(2*H,2*H)

        # Parameters of Classification Layer
        self.content = nn.Linear(2*H,1,bias=False)
        self.salience = nn.Bilinear(2*H,2*H,1,bias=False)
        self.novelty = nn.Bilinear(2*H,2*H,1,bias=False)
        self.abs_pos = nn.Linear(P_D,1,bias=False)
        self.rel_pos = nn.Linear(P_D,1,bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1,0.1))
    def max_pool1d(self,x,seq_lens):
        # x:[N,L,O_in]
        out = []
        for index,t in enumerate(x):
            try:
                t = t[:seq_lens[index],:]
            except:
                print(index)
                print(seq_lens[index])
                print(t.shape)
                t = t[:seq_lens[index],:]
            t = torch.t(t).unsqueeze(0)
            out.append(F.max_pool1d(t,t.size(2)))

        out = torch.cat(out).squeeze(2)
        return out
    def forward(self,x,doc_lens):
        N = x.size(0)
        L = x.size(1)
        B = int(N/self.args.pos_num) 
        H = self.args.hidden_size
        word_mask = torch.ones_like(x) - torch.sign(x)
        word_mask = word_mask.data.type(torch.cuda.ByteTensor).view(N,1,L)
        
        x = self.embed(x)                                # (N,L,D)
        x,_ = self.word_RNN(x)
        
        # attention
        query = self.word_query.expand(N,-1,-1).contiguous()
        self.attn.set_mask(word_mask)
        word_out, word_scores = self.attn(query,x)
        #print(word_scores.shape)
        word_out = word_out.squeeze(1)      # (N,2*H)
        word_scores = word_scores.squeeze(1)
        #print(word_scores.shape)
        x = self.pad_doc(word_out,doc_lens)
        # sent level GRU
        sent_out = self.sent_RNN(x)[0]                                           # (B,max_doc_len,2*H)
        #docs = self.avg_pool1d(sent_out,doc_lens)                               # (B,2*H)
        max_doc_len = max(doc_lens)
        docs = self.max_pool1d(sent_out,doc_lens)
        probs = []
        
        for index in range(0,B):
            doc_len = self.args.pos_num
            valid_hidden = sent_out[index,:doc_len,:]                            # (doc_len,2*H)
            doc = F.tanh(self.fc(docs[index])).unsqueeze(0)
            doc = self.dropout(doc)
            s = Variable(torch.zeros(1,2*H))
            if self.args.device is not None:
                s = s.cuda()
            for position, h in enumerate(valid_hidden):
                h = h.view(1, -1)                                                # (1,2*H)
                # get position embeddings
                abs_index = Variable(torch.LongTensor([[position]]))
                if self.args.device is not None:
                    abs_index = abs_index.cuda()
                abs_features = self.abs_pos_embed(abs_index).squeeze(0)
                
                rel_index = int(round((position + 1) * 9.0 / doc_len))
                rel_index = Variable(torch.LongTensor([[rel_index]]))
                if self.args.device is not None:
                    rel_index = rel_index.cuda()
                rel_features = self.rel_pos_embed(rel_index).squeeze(0)
                
                # classification layer
                content = self.content(h) 
                content = self.dropout(content)
                salience = self.salience(h,doc)
                salience = self.dropout(salience)
                novelty = -1 * self.novelty(h,F.tanh(s))
                novelty = self.dropout(novelty)
                abs_p = self.abs_pos(abs_features)
                rel_p = self.rel_pos(rel_features)
                prob = F.sigmoid(content + salience + novelty + abs_p + rel_p + self.bias)
                s = s + torch.mm(prob,h)
                #print position,F.sigmoid(abs_p + rel_p)
                probs.append(prob)
        return torch.cat(probs).squeeze(), word_scores
