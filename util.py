import torch
from phe import paillier
import numpy as np

def MVNP(n):
    public_key, private_key = paillier.generate_paillier_keypair()
    enc_r=[]
    for i in range(n):
        enc_r.append([public_key.encrypt(x) for x in np.random.randint(100,size=1).tolist()])
    enc_R=np.sum(enc_r,0)
    R=[int(private_key.decrypt(x)) for x in enc_R]
    return R
def to_mask(w,r):
    value=w+r
    return value
def from_w_to_list(w):
    value = {}
    for key in w.keys():
        value[key] = w[key].cpu().tolist()
    return value

def from_w_to_tensor(w):
    value = torch.Tensor()
    for key in w.keys():
        value=w[key].view(-1) if not len(value) else torch.cat([value,w[key].view(-1)])
    return value

def from_tensor_to_list(t):
    return t.cpu().tolist()

def from_w_to_zkp(w):
    value_lists = []
    for key in w.keys():
        if "bias" in key:
            value_lists.append((((w[key].unsqueeze(0))).tolist()))
            # break
        else:
            # continue
            if "conv" in key:
                value_lists.append((w[key].reshape(w[key].size()[0],-1)).tolist())
            else:
                value_lists.append(((w[key]).tolist()))
    return value_lists

def from_list_to_w(w):
    value = {}
    for key in w.keys():
        value[key] = torch.Tensor(w[key])
    return value