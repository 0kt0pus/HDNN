from utils import *
from ops import *
import pandas as pd
import numpy as np

hd_size = 10000
# load mnist
a = pd.read_csv('./data/mnist_train.csv', header=0)
b = a.to_numpy()[:, 1:]
label = a.to_numpy()[:, 0]
#print(b.shape)
img_batch = torch.from_numpy((b > 112).astype(np.int)).int()[0:100]
#print(img_batch.shape)
encoder = make_sparse_bin_tensor(img_batch.shape[1], hd_size)
cls = make_sparse_bin_tensor(2, hd_size)
#print(encoder)
print("")

encoded_batch = torch.mm(img_batch, encoder)
## apply majority rule to each image hd vect
encoded_thresholded_batch = torch.zeros((img_batch.shape[0], hd_size), dtype=torch.int)
#print((encoded_batch[0] > torch.max(encoded_batch[0] /2)).int())
#print((encoded_batch[11] > torch.max(encoded_batch[11] /2)).int())
#print(label[0])
#print(label[11])
for i in range(img_batch.shape[0]):
    hyv = encoded_batch[i]
    threshold = torch.min(hyv) + ((torch.max(hyv) - torch.min(hyv)) / 2)
    hyv = (hyv > threshold).int()
    encoded_thresholded_batch[i] = hyv

img_hyv_1 = encoded_thresholded_batch[0]
img_hyv_2 = encoded_thresholded_batch[11]
img_hyv_3 = encoded_thresholded_batch[35]
img_hyv_4 = encoded_thresholded_batch[1]
img_hyv_44 = encoded_thresholded_batch[21]
img_hyv_444 = encoded_thresholded_batch[34]
img_hyv_5 = encoded_thresholded_batch[7]

#inv_img_hyv_3 = (img_hyv_3 < 0.5).int()
#inv_img_hyv_4 = (img_hyv_4 < 0.5).int()

cls_1 = cls[0]
print(cls_1)
print("")
#inv_cls_1 = (cls_1 < 0.5).int()
cls_2 = cls[1]
#inv_cls_2 = (cls_2 < 0.5).int()

## Train
img_hyv_cl_1 = bind(cls_1, img_hyv_1)
img_hyv_cl_2 = bind(cls_1, img_hyv_2)
concept_5 = bundle(img_hyv_cl_1, img_hyv_cl_2)

img_hyv_cl_4 = bind(cls_2, img_hyv_4)
img_hyv_cl_44 = bind(cls_2, img_hyv_44)
concept_0 = bundle(img_hyv_cl_4, img_hyv_cl_44)
'''
## Similaritiy presevation in XOR
dist = hamming_distance(img_hyv_cl_1, img_hyv_cl_2)
print(dist)
dist = hamming_distance(img_hyv_1, img_hyv_2)
print(dist)
'''

## Test
## similar images have low hamming distance
query = bind(cls_1, concept_5)
dist_a = hamming_distance(img_hyv_3, query)
dist_c = hamming_distance(img_hyv_2, query)
dist_b = hamming_distance(img_hyv_4, query)
dist_d = hamming_distance(img_hyv_5, query)
print(dist_a)
print(dist_c)
print(dist_b)
print(dist_d)
print("")

## similar classes have small hamming distance
query = bind(img_hyv_3, concept_5)
dist_a = hamming_distance(cls_1, query)
dist_c = hamming_distance(cls_2, query)
print(dist_a)
print(dist_c)
print("")

query_5 = bind(img_hyv_4, concept_5)
query_0 = bind(img_hyv_4, concept_0)
dist_b = hamming_distance(cls_1, query_5)
dist_d = hamming_distance(cls_2, query_0)

print(dist_b)
print(dist_d)
print("")

query_55 = bind(img_hyv_444, concept_5)
query_00 = bind(img_hyv_444, concept_0)
dist_b = hamming_distance(cls_1, query_55)
dist_d = hamming_distance(cls_2, query_00)

print(dist_b)
print(dist_d)
print("")