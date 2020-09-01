from utils import *
from ops import *
import pandas as pd
import numpy as np

hd_size = 50
# load mnist
a = pd.read_csv('./data/mnist_train.csv', header=0)
b = a.to_numpy()[:, 1:]
#print(b.shape)
img_batch = torch.from_numpy((b > 112).astype(np.int)).int()
encoder = make_sparse_bin_tensor(2, hd_size)
print(encoder)
print("")

def encode_binary_features(img, feature_hyv, hd_size):
    img_hyv = torch.zeros(img.shape[0], hd_size)
    #img_vect = torch.squeeze(img, 0)
    for i in range(img.shape[0]):
        if img[i] == 0:
            img_hyv[i] = feature_hyv[0]
        else:
            img_hyv[i] = feature_hyv[1]

    ## sum along the feature axis and threshold the values
    img_hyv = torch.sum(img_hyv, 0)
    print(img_hyv)
    threshold = torch.min(img_hyv) + ((torch.max(img_hyv) - torch.min(img_hyv)) / 2)
    return (img_hyv > threshold).int()

##print(encode_binary_features(img, encoder, hd_size))

def encode_batch(img_batch, feature_hyv, hd_size):
    ## iterate take image and encode it and insert to a tensor
    batch_size = img_batch.shape[0]
    encoded_batch = torch.zeros(batch_size, hd_size)
    for i in range(batch_size):
        img = img_batch[i]
        #print(img.shape)
        ## encode the image
        encoded_img = encode_binary_features(img, feature_hyv, hd_size)
        #print(encoded_img)
        #encode_batch[i] = encoded_img

    return encode_batch

encode_batch(img_batch, encoder, hd_size)


