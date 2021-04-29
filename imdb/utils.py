import math
import pickle
import numpy as np
import torch


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def logistic_anneal(step, a=0.0025, x0=2500):
    return 1. / (1. + math.exp(-a * (step - x0)))

def linear_anneal(step, x0, initial=0.01):
    return min(1., initial + step / x0)

def interpolate(start, end, num):
    res = np.zeros([num, start.shape[0]])
    for dim, (s, e) in enumerate(zip(start, end)):
        res[:, dim] = np.linspace(s, e, num)
    return res

def transform(idxs, idx_to_word, eos_idx):
    sents = []
    for sent in idxs:
        tmp = []
        for idx in sent:
            if idx == eos_idx:
                break
            tmp.append(idx_to_word[idx])
        sents.append(' '.join(tmp))

    return sents

def log_Normal_diag(x, mean, log_var, dim=None):
    log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) * torch.pow( torch.exp( log_var ), -1) )
    

    return torch.sum( log_normal, dim )

def log_Normal_standard(x, dim=None):
    log_normal = -0.5 * torch.pow( x , 2 )
    
    return torch.sum( log_normal, dim )
