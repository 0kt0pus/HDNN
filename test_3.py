from utils import *
import ops
import pandas as pd
import numpy as np

HD_SIZE = 10000
NUM_CLASSES = 10
NUM_IMG = 100
IMG_SIZE = 28 * 28
PIX_THRESHOLD = int(255 / 2)

# load mnist
a = pd.read_csv('./data/mnist_train.csv', header=0)
b = a.to_numpy()[:, 1:]
label = a.to_numpy()[:, 0]
#print(b.shape)
img_batch = torch.from_numpy((b > PIX_THRESHOLD).astype(np.int)).int()[0:NUM_IMG]
label_batch = label[0:NUM_IMG]
#print(label_batch[0:10])
#print(img_batch.shape)
encode_tensor = make_sparse_bin_tensor(IMG_SIZE, HD_SIZE)
cls_tensor = make_sparse_bin_tensor(NUM_CLASSES, HD_SIZE)

img_batch_hd_vec = dot_encoder(img_batch, encode_tensor)
## Iterate bind bundle and generate the class prototypes / Concpets
# A tensor to hold class prototypes / concepts
concept_tensor = torch.zeros((NUM_CLASSES, HD_SIZE), dtype=torch.int)

for i in range(NUM_IMG):
    ## get the image hypervect
    img_hd_vec = img_batch_hd_vec[i]
    ## get the corresponding class
    cls = label_batch[i]
    ## Bind the image hypervect with class hypervect
    img_cls_hd_vec = ops.bind(img_hd_vec, cls_tensor[cls])
    ## bundle with other prototypes of the same class to make
    ## a complete class prototype
    concept_tensor[cls] = ops.bundle(img_cls_hd_vec, concept_tensor[cls])

#complete_concept = ops.self_bundle(concept_tensor)

## Test on train images
#print(concept_tensor)
# Take the train image at 0th index which is cls 5
img_hd_vec_5 = img_batch_hd_vec[0]
#print((img_hd_vec_5).type())
#print((concept_tensor).type())
## Query the concept tensor with the image hd tensor
query = ops.bind(img_hd_vec_5, concept_tensor)
#query_complete = ops.bind(img_hd_vec_5, complete_concept)

#print(type(query))
## Now check for each class hamming distance
dist_dict = dict()
for i in range(NUM_CLASSES):
    cls_hd_vec = cls_tensor[i]
    hamm_dist = ops.hamming_distance(cls_hd_vec, query)
    dist_dict[i] = hamm_dist
    print("class {}: distance {}".format(i, hamm_dist))

print("Test on Train class: {}, actural class {}".format(min(dist_dict, key=dist_dict.get), label[0]))
print("")
## Test on testing data
a = pd.read_csv('./data/mnist_test.csv', header=0)
b = a.to_numpy()[:, 1:]
label_test = a.to_numpy()[:, 0]
img_batch_test = torch.from_numpy((b > PIX_THRESHOLD).astype(np.int)).int()[0:20]
img_batch_test_hd_vec = dot_encoder(img_batch_test, encode_tensor)

img_test_hd_vec_5 = img_batch_test_hd_vec[8]
#print(label_test[15])
query = ops.bind(img_test_hd_vec_5, concept_tensor)

## Now check for each class hamming distance
dist_dict = dict()
for i in range(NUM_CLASSES):
    cls_hd_vec = cls_tensor[i]
    hamm_dist = ops.hamming_distance(cls_hd_vec, query)
    dist_dict[i] = hamm_dist
    #print("class {}: distance {}".format(i, hamm_dist))

print("Test on Test class: {}, actural class {}".format(min(dist_dict, key=dist_dict.get), label_test[8]))

print("")
img_test_hd_vec_5 = img_batch_test_hd_vec[15]
#print(label_test[15])
query = ops.bind(img_test_hd_vec_5, concept_tensor)

## Now check for each class hamming distance
dist_dict = dict()
for i in range(NUM_CLASSES):
    cls_hd_vec = cls_tensor[i]
    hamm_dist = ops.hamming_distance(cls_hd_vec, query)
    dist_dict[i] = hamm_dist
    #print("class {}: distance {}".format(i, hamm_dist))

print("Test on Test class: {}, actural class {}".format(min(dist_dict, key=dist_dict.get), label_test[15]))
