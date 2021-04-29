"""
Copyright 2018 NAVER Corp.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import os
import numpy as np
import random
import sys
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from helper import gVar, gData
from modules import Encoder, ContextEncoder, Variation, Decoder, mean_zero_Variation           
import flows as flow
import nn as nn_
one = gData(torch.FloatTensor([1]))
minus_one = one * -1    
def log_Normal_diag(x, mean, log_var, average=True, dim=1):
    log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) * torch.pow( torch.exp( log_var ), -1) )
    if average:
        return torch.mean( log_normal, dim )
    else:
        return torch.sum( log_normal, dim )
class KLPFCVAE(nn.Module):
    def __init__(self, config, vocab_size, PAD_token=0):
        super(KLPFCVAE, self).__init__()
        self.vocab_size = vocab_size
        self.maxlen=config['maxlen']
        self.clip = config['clip']
        self.lambda_gp = config['lambda_gp']
        self.temp=config['temp']
        
        self.embedder= nn.Embedding(vocab_size, config['emb_size'], padding_idx=PAD_token)
        self.utt_encoder = Encoder(self.embedder, config['emb_size'], config['n_hidden'], 
                                   True, config['n_layers'], config['noise_radius']) 
        self.context_encoder = ContextEncoder(self.utt_encoder, config['n_hidden']*2+2, config['n_hidden'], 1, config['noise_radius']) 
        self.prior_net = mean_zero_Variation(config['n_hidden'], config['z_size']) # p(e|c)
        self.post_net = Variation(config['n_hidden']*3, config['z_size']) # q(e|c,x)
        
        self.postflow1 = flow.myIAF(config['z_size'],config['z_size'] * 2, config['n_hidden'],3)
        self.priorflow1 = flow.IAF(config['z_size'],config['z_size'] * 2, config['n_hidden'],3)
        self.priorflow2 = flow.IAF(config['z_size'],config['z_size'] * 2, config['n_hidden'],3)
        self.priorflow3 = flow.IAF(config['z_size'],config['z_size'] * 2, config['n_hidden'],3)
        
        self.post_generator = self.postflow1
        self.prior_generator = nn_.SequentialFlow(self.priorflow1,self.priorflow2,self.priorflow3)
                                                                                             
        self.decoder = Decoder(self.embedder, config['emb_size'], config['n_hidden']+config['z_size'], 
                               vocab_size, n_layers=1) 
           
        self.optimizer_AE = optim.SGD(list(self.context_encoder.parameters())
                                      +list(self.post_net.parameters())
                                      +list(self.post_generator.parameters())
                                      +list(self.decoder.parameters())
                                      +list(self.prior_net.parameters())
                                      +list(self.prior_generator.parameters())
                                      ,lr=config['lr_ae'])
        self.optimizer_G = optim.RMSprop(list(self.post_net.parameters())
                                      +list(self.post_generator.parameters())
                                      +list(self.prior_net.parameters())
                                      +list(self.prior_generator.parameters())
                                      , lr=config['lr_gan_g'])
        
        
        self.lr_scheduler_AE = optim.lr_scheduler.StepLR(self.optimizer_AE, step_size = 10, gamma=0.6)
        
        self.criterion_ce = nn.CrossEntropyLoss()
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):        
            m.weight.data.uniform_(-0.02, 0.02)
            m.bias.data.fill_(0)
    def sample_post(self, x, c):
        xc = torch.cat((x, c),1)
        e, mu, log_s = self.post_net(xc)
        z, det_f,_,_ = self.post_generator((e,torch.eye(e.shape[1]), c, mu))
        return z, mu, log_s, det_f
    def sample_code_post(self, x, c):
        xc = torch.cat((x, c),1)
        e, mu, log_s = self.post_net(xc)
        z, det_f,_,_ = self.post_generator((e, torch.eye(e.shape[1]), c, mu))
        return z, mu, log_s, det_f
    def sample_post2(self, x, c):
        xc = torch.cat((x, c),1)
        e, mu, log_s = self.post_net(xc)
        z, det_f,_,_ = self.post_generator((e, torch.eye(e.shape[1]), c, mu))
        return e, mu, log_s, z , det_f
   
    def sample_code_prior(self, c):
        e, mu, log_s = self.prior_net(c)
        return e, mu, log_s  
    def sample_prior(self,c):
        e, mu, log_s = self.prior_net(c)
        return e
    def train_AE(self, context, context_lens, utt_lens, floors, response, res_lens):
        self.context_encoder.train()
        self.decoder.train()
        c = self.context_encoder(context, context_lens, utt_lens, floors)
        x,_ = self.utt_encoder(response[:,1:], res_lens-1)      
        z_post, mu_post, log_s_post, weight = self.sample_code_post(x, c)
        output = self.decoder(torch.cat((z_post, c),1), None, response[:,:-1], (res_lens-1))  
        flattened_output = output.view(-1, self.vocab_size) 
        
        dec_target = response[:,1:].contiguous().view(-1)
        mask = dec_target.gt(0) # [(batch_sz*seq_len)]
        masked_target = dec_target.masked_select(mask) # 
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)# [(batch_sz*seq_len) x n_tokens]
        masked_output = flattened_output.masked_select(output_mask).view(-1, self.vocab_size)
        self.optimizer_AE.zero_grad()
        AE_term = self.criterion_ce(masked_output/self.temp, masked_target)
        loss = AE_term
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(list(self.context_encoder.parameters())+list(self.decoder.parameters())+list(self.post_generator.parameters())+list(self.prior_generator.parameters())+list(self.post_net.parameters())+list(self.prior_net.parameters()), self.clip)
        self.optimizer_AE.step()

        return [('train_loss_AE', AE_term.item())]#,('KL_loss', KL_loss.mean().item())]#,('det_f', det_f.mean().item()),('det_g', det_g.mean().item())]        
    
    def train_G(self, context, context_lens, utt_lens, floors, response, res_lens): 
        self.context_encoder.eval()
        self.optimizer_G.zero_grad()
        c = self.context_encoder(context, context_lens, utt_lens, floors)
        # -----------------posterior samples ---------------------------
        x,_ = self.utt_encoder(response[:,1:], res_lens-1)
        z_0, mu_post, log_s_post, z_post, weight = self.sample_post2(x.detach(), c.detach())
        # ----------------- prior samples ---------------------------
        prior_z, mu_prior, log_s_prior = self.sample_code_prior(c.detach())
        KL_loss = torch.sum(log_s_prior - log_s_post + torch.exp(log_s_post)/torch.exp(log_s_prior) * torch.sum(weight**2,dim=2) + (mu_post)**2/torch.exp(log_s_prior),1) / 2 - 100 
        loss = KL_loss 
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(list(self.post_generator.parameters())+list(self.prior_generator.parameters())+list(self.post_net.parameters())+list(self.prior_generator.parameters()), self.clip)
        self.optimizer_G.step()
        return [('KL_loss', KL_loss.mean().item())]#,('det_f', det_f.mean().item()),('det_g', det_g.sum().item())]
    
    
    def valid(self, context, context_lens, utt_lens, floors, response, res_lens):
        self.context_encoder.eval()      
        self.decoder.eval()
        
        c = self.context_encoder(context, context_lens, utt_lens, floors)
        x,_ = self.utt_encoder(response[:,1:], res_lens-1)
        post_z, mu_post, log_s_post, det_f = self.sample_code_post(x, c)
        prior_z, mu_prior, log_s_prior = self.sample_code_prior(c)
        KL_loss = torch.sum(log_s_prior - log_s_post + (torch.exp(log_s_post) + (mu_post)**2)/torch.exp(log_s_prior),1) / 2
        loss =  KL_loss
        costG = loss.sum()
        dec_target = response[:,1:].contiguous().view(-1)
        mask = dec_target.gt(0) # [(batch_sz*seq_len)]
        masked_target = dec_target.masked_select(mask) 
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)
        output = self.decoder(torch.cat((post_z, c),1), None, response[:,:-1], (res_lens-1)) 
        flattened_output = output.view(-1, self.vocab_size) 
        masked_output = flattened_output.masked_select(output_mask).view(-1, self.vocab_size)
        lossAE = self.criterion_ce(masked_output/self.temp, masked_target)
        return [('valid_loss_AE', lossAE.item()),('valid_loss_G', costG.item())]
    
    def sample(self, context, context_lens, utt_lens, floors, repeat, SOS_tok, EOS_tok):    
        self.context_encoder.eval()
        self.decoder.eval()
        
        c = self.context_encoder(context, context_lens, utt_lens, floors)
        c_repeated = c.expand(repeat, -1)
        prior_z = self.sample_prior(c_repeated)    
        sample_words, sample_lens= self.decoder.sampling(torch.cat((prior_z,c_repeated),1), 
                                                         None, self.maxlen, SOS_tok, EOS_tok, "greedy") 
        return sample_words, sample_lens
    def gen(self, context, prior_z, context_lens, utt_lens, floors, repeat, SOS_tok, EOS_tok):
        self.context_encoder.eval()
        self.decoder.eval()
        c = self.context_encoder(context, context_lens, utt_lens, floors)
        c_repeated = c.expand(repeat, -1)
        sample_words, sample_lens= self.decoder.sampling(torch.cat((prior_z,c_repeated),1), 
                                                         None, self.maxlen, SOS_tok, EOS_tok, "greedy")
        return sample_words ,sample_lens
    def sample_latent(self, context, context_lens, utt_lens, floors, repeat, SOS_tok, EOS_tok):
        self.context_encoder.eval()
        #self.decoder.eval()
        c = self.context_encoder(context, context_lens, utt_lens, floors)
        c_repeated = c.expand(repeat, -1)
        e,_,_ = self.sample_code_prior(c_repeated)
        prior_z, _ , _ = self.prior_generator((e,0, c_repeated))
        return prior_z ,e
    def sample_latent_post(self, context, context_lens, utt_lens, floors, response, res_lens,repeat):
        self.context_encoder.eval()
        c = self.context_encoder(context, context_lens, utt_lens, floors)
        x,_ = self.utt_encoder(response[:,1:], res_lens-1)
        c_repeated = c.expand(repeat, -1)
        x_repeated = x.expand(repeat, -1)
        z, mu_post, log_s_post, det_f = self.sample_post(x_repeated, c_repeated)
        z_post = z
        return z_post,z
    def adjust_lr(self):
        self.lr_scheduler_AE.step()
    


