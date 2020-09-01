import torch
import ops

def make_sparse_bin_tensor(num_features, hd_size):
    a = torch.rand((num_features, hd_size))
    #mid = (torch.max(a) - torch.min(a)) / 2.0
    return (a > (torch.max(a) - 0.1)).int()

def dot_encoder(img_batch, encode_tensor):
    encoded_batch = torch.mm(img_batch, encode_tensor)
    num_img = img_batch.shape[0]
    hd_size = encode_tensor.shape[1]
    ## apply majority rule to each image hd vect
    encoded_thresholded_batch = torch.zeros((num_img, hd_size), dtype=torch.int)
    #print((encoded_batch[0] > torch.max(encoded_batch[0] /2)).int())
    #print((encoded_batch[11] > torch.max(encoded_batch[11] /2)).int())
    #print(label[0])
    #print(label[11])
    for i in range(num_img):
        hyv = encoded_batch[i]
        threshold = torch.min(hyv) + ((torch.max(hyv) - torch.min(hyv)) / 2)
        hyv = (hyv > threshold).int()
        encoded_thresholded_batch[i] = hyv

    return encoded_thresholded_batch

'''
a = torch.rand(3, 3) > 0.5
b = torch.rand(3, 3) > 0.5
print(a)
print("")
print(b)
print("")
print(bind(a, b))
print("")
print(bundle(a, b))
'''