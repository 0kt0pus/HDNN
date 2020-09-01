import torch

def bind(lhs, rhs):
    return lhs ^ rhs

def bundle(lhs, rhs):
    combined = lhs.int() + rhs.int()
    mid = (torch.max(combined) - torch.min(combined)) / 2
    #print(combined)
    return (combined > (torch.min(combined) + mid)).int()

def self_bundle(lhs):
    combined = torch.sum(lhs.int(), 0)
    mid = (torch.max(combined) - torch.min(combined)) / 2
    #print(combined)
    return combined > (torch.min(combined) + mid)

def hamming_distance(lhs, rhs):
    xor_vect = bind(lhs, rhs)
    return torch.sum(xor_vect)
