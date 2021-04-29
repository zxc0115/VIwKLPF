
# coding: utf-8

# In[2]:


import sys
sys.path.append('../')

import os
import time
import datetime
import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.manifold import TSNE
from multiprocessing import cpu_count

from imdb_data import IMDB
from KLPF.KLPF_model import KLPF
from utils import linear_anneal, log_Normal_diag, log_Normal_standard


# In[3]:


# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

max_len = 80
batch_size = 32
splits = ['train', 'valid', 'test']

# Penn TreeBank (PTB) dataset
data_path = '../data'
datasets = {split: IMDB(root=data_path, split=split) for split in splits}

# dataloader
dataloaders = {split: DataLoader(datasets[split],
                                    batch_size=batch_size,
                                    shuffle=split=='train',
                                    num_workers=cpu_count(),
                                    pin_memory=torch.cuda.is_available())
                                    for split in splits}

symbols = datasets['train'].symbols


# In[4]:


len(datasets['train'])


# In[5]:


# iaf model
embedding_size = 300
hidden_size = 256
latent_dim = 32
dropout_rate = 0.5
model = KLPF(vocab_size=datasets['train'].vocab_size,
               embed_size=embedding_size,
               time_step=max_len,
               hidden_size=hidden_size,
               z_dim=latent_dim,
               dropout_rate=dropout_rate,
               bos_idx=symbols['<bos>'],
               eos_idx=symbols['<eos>'],
               pad_idx=symbols['<pad>'],
               n_comb = 2)
model = model.to(device)


# In[6]:


# folder to save model
if False:
    save_path = 'iaf'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


# In[7]:


# objective function
learning_rate = 0.001
criterion = nn.NLLLoss(size_average=False, ignore_index=symbols['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

criterion_c = nn.CrossEntropyLoss()

# negative log likelihood
def NLL(logp, target, length):
    target = target[:, :torch.max(length).item()].contiguous().view(-1)
    logp = logp.view(-1, logp.size(-1))
    return criterion(logp, target)

# KL divergence
def KL_div(mu, logvar):
    return -0.5 * torch.sum(1. + logvar - mu.pow(2) - logvar.exp())


# In[8]:


# training setting
epoch = 20
print_every = 50

# training interface
step = 0
tracker = {'ELBO': [], 'NLL': [], 'KL': [], 'KL_weight': [], 'accuracy': []}
start_time = time.time()
for ep in range(epoch):
    # learning rate decay
    if ep >= 10 and ep % 2 == 0:
        learning_rate = learning_rate * 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    for split in splits:
        dataloader = dataloaders[split]
        model.train() if split == 'train' else model.eval()
        totals = {'ELBO': 0., 'NLL': 0., 'KL': 0., 'words': 0, 'correct': 0}
        for itr, (enc_inputs, dec_inputs, targets, lengths, labels) in enumerate(dataloader):
            bsize = enc_inputs.size(0)
            enc_inputs = enc_inputs.to(device)
            dec_inputs = dec_inputs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            # forward
            logp, z_0, z_T, mu, logvar, logc, labels = model(enc_inputs, dec_inputs, lengths, labels)
            

            # calculate loss
            NLL_loss = NLL(logp, targets, lengths + 1)
            # classification loss
            #class_loss = nn.CrossEntropyLoss(logc, labels)
            class_loss = criterion_c(logc, labels)
            
            
            # KL loss
            log_p_z = log_Normal_standard(z_T, dim=1)
            log_q_z = log_Normal_diag(z_0, mu, logvar, dim=1)
            KL_loss = torch.sum(-(log_p_z - log_q_z))
            KL_weight = linear_anneal(step, len(dataloaders['train']) * 10)
            loss = (NLL_loss + KL_weight * KL_loss) / bsize + class_loss         
            
            
            # cumulate
            totals['ELBO'] += loss.item() * bsize
            totals['NLL'] += NLL_loss.item()
            totals['KL'] += KL_loss.item()
            totals['words'] += torch.sum(lengths).item()
            _, predicted = torch.max(logc, -1)
            correct = (predicted == labels).sum().item()
            totals['correct'] += correct

            # backward and optimize
            if split == 'train':
                step += 1
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

                # track
                tracker['ELBO'].append(loss.item())
                tracker['NLL'].append(NLL_loss.item() / bsize)
                tracker['KL'].append(KL_loss.item() / bsize)
                tracker['KL_weight'].append(KL_weight)
                
                if(False):
                    # print statistics
                    if itr % print_every == 0 or itr + 1 == len(dataloader):
                        print("%s Batch %04d/%04d, ELBO-Loss %.4f, "
                              "NLL-Loss %.4f, KL-Loss %.4f, KL-Weight %.4f,"
                              "Loss %.4f, Accuracy %2.2f"
                              % (split.upper(), itr, len(dataloader),
                                 tracker['ELBO'][-1], tracker['NLL'][-1],
                                 tracker['KL'][-1], tracker['KL_weight'][-1],
                                 class_loss, correct / bsize * 100))
            #break
        #break

        samples = len(datasets[split])
        print("%s Epoch %02d/%02d, ELBO %.4f, NLL %.4f, KL %.4f, PPL %.4f, Accuracy %2.2f"
              % (split.upper(), ep, epoch, totals['ELBO'] / samples,
                 totals['NLL'] / samples, totals['KL'] / samples,
                 math.exp(totals['NLL'] / totals['words']), totals['correct'] / samples * 100))
    print('\n')

    #break
    # save checkpoint
    #checkpoint_path = os.path.join(save_path, "E%02d.pkl" % ep)
    #torch.save(model.state_dict(), checkpoint_path)
    #print("Model saved at %s\n" % checkpoint_path)
end_time = time.time()
print('Total cost time',
      time.strftime("%H hr %M min %S sec", time.gmtime(end_time - start_time)))


# In[9]:


# another KL

totals = {'ELBO': 0., 'NLL': 0., 'KL': 0., 'words': 0}
with torch.no_grad():
    model.eval()
    for itr, (enc_inputs, dec_inputs, targets, lengths, labels) in enumerate(dataloaders['test']):
        bsize = enc_inputs.size(0)
        enc_inputs = enc_inputs.to(device)
        dec_inputs = dec_inputs.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        # forward
        logp, z_0, z_T, mu, logvar, _, _ = model(enc_inputs, dec_inputs, lengths, labels)

        # calculate loss
        NLL_loss = NLL(logp, targets, lengths + 1)
        # KL loss
        log_p_z = log_Normal_standard(z_0, dim=1)
        log_q_z = log_Normal_diag(z_0, mu, logvar, dim=1)
        KL_loss = torch.sum(log_q_z - log_p_z)
        #KL_weight = linear_anneal(step, len(dataloaders['train']) * 10)
        KL_weight = 1
        loss = (NLL_loss + KL_weight * KL_loss) / bsize

        # cumulate
        totals['ELBO'] += loss.item() * bsize
        totals['NLL'] += NLL_loss.item()
        totals['KL'] += KL_loss.item()
        totals['words'] += torch.sum(lengths).item()
    samples = len(datasets['test'])
    print("%s Epoch %02d/%02d, ELBO %.4f, NLL %.4f, KL %.4f, PPL %.4f"
          % (split.upper(), ep, epoch, totals['ELBO'] / samples,
             totals['NLL'] / samples, totals['KL'] / samples,
             math.exp(totals['NLL'] / totals['words'])))


# In[10]:


# calculate au

delta = 0.01
with torch.no_grad():
    model.eval()
    cnt = 0
    for itr, (enc_inputs, dec_inputs, targets, lengths, labels) in enumerate(dataloaders['test']):
        bsize = enc_inputs.size(0)
        enc_inputs = enc_inputs.to(device)
        dec_inputs = dec_inputs.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        # forward
        logp, z_0, z_T, mu, logvar, _, _ = model(enc_inputs, dec_inputs, lengths, labels)
        
        if cnt == 0:
            mu_sum = mu.sum(dim=0, keepdim=True)
        else:
            mu_sum = mu_sum + mu.sum(dim=0, keepdim=True)
        cnt += mu.size(0)
        
    mu_mean = mu_sum / cnt
        
    cnt = 0
    for itr, (enc_inputs, dec_inputs, targets, lengths, labels) in enumerate(dataloaders['test']):
        bsize = enc_inputs.size(0)
        enc_inputs = enc_inputs.to(device)
        dec_inputs = dec_inputs.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        # forward
        logp, z_0, z_T, mu, logvar, _, _ = model(enc_inputs, dec_inputs, lengths, labels)
        
        if cnt == 0:
            var_sum = ((mu - mu_mean) ** 2).sum(dim=0)
        else:
            var_sum = var_sum + ((mu - mu_mean) ** 2).sum(dim=0)
        cnt += mu.size(0)
        
    au_var = var_sum / (cnt - 1)
    
    print((au_var >= delta).sum().item())
    #print(au_var)


# In[11]:


# plot KL curve
if(True):
    fig, ax1 = plt.subplots()
    lns1 = ax1.plot(tracker['KL_weight'], 'b', label='KL term weight')
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('Step')
    ax1.set_ylabel('KL term weight')
    ax2 = ax1.twinx()
    lns2 = ax2.plot(tracker['KL'], 'r', label='KL term value')
    ax2.set_ylabel('KL term value')
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, bbox_to_anchor=(0., 1.02, 1., .102),
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


# In[12]:


# latent space visualization
if(True):
    features = np.empty([len(datasets['test']), latent_dim])
    feat_label = np.empty(len(datasets['test']))
    for itr, (enc_inputs, dec_inputs, _, lengths, labels) in enumerate(dataloaders['test']):
        enc_inputs = enc_inputs.to(device)
        dec_inputs = dec_inputs.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        _, _, _, mu, _, _, labels = model(enc_inputs, dec_inputs, lengths, labels)
        start, end = batch_size * itr, batch_size * (itr + 1)
        features[start:end] = mu.data.cpu().numpy()
        feat_label[start:end] = labels.data.cpu().numpy()

    


# In[13]:


tsne_z = TSNE(n_components=2,perplexity=100).fit_transform(features)
tracker['z'] = tsne_z
tracker['label'] = feat_label


# In[14]:


plt.figure()
for i in range(2):
    plt.scatter(tsne_z[np.where(feat_label == i), 0], tsne_z[np.where(feat_label == i), 1], s=10, alpha=0.5)
plt.show()


# In[15]:


# save learning results
sio.savemat("KLPF.mat", tracker)

