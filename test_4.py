import torch
import torch.nn as nn
import torch.optim as optim

import ops
from utils import *

import pandas as pd
import numpy as np

HD_SIZE = 10000
HIDDEN_1 = 500
NUM_CLASSES = 10
NUM_IMG_TRAIN = 1000
IMG_SIZE = 28 * 28
PIX_THRESHOLD = int(255 / 2)

EPOCH = 100
LR = 0.01
# load mnist
a = pd.read_csv('./data/mnist_train.csv', header=0)
b = a.to_numpy()[:, 1:]
label = torch.from_numpy(a.to_numpy()[:, 0])[0:NUM_IMG_TRAIN]
#print(b.shape)
img_batch = torch.from_numpy((b > PIX_THRESHOLD).astype(np.int)).float()[0:NUM_IMG_TRAIN]
label_batch = label[0:NUM_IMG_TRAIN]

def binarize_outputs(x):
    x_mid = (torch.max(x) - torch.min(x)) / 2.0
    ##return (x > (torch.min(x) + x_mid)).int()
    return (x > (torch.max(x) - 0.1)).int()

def make_img_tensor(img_tensor):
    return img_tensor.unsqueeze(0)

def make_one_hot(preds, num_classes):
    one_hot_tensor = torch.zeros((len(preds), num_classes))#, requires_grad=True)
    for i, pred in enumerate(preds):
        #print(pred)
        one_hot_tensor[i, pred] = 1
    
    return one_hot_tensor

def hd_predictor(pred_encode_list, concept_tensor, cls_tensor):
    ## collect the lowest hamming distance for each image
    pred_lbl_list = []
    ## each img hd vec query each concept
    for i in range(len(pred_encode_list)):
        pred_encode = pred_encode_list[i]
        ## put the class and distance into a dict
        dist_dict = dict()
        ## query each concept with the encoded vector to
        ## get a noisy prediction of the class
        for j in range(NUM_CLASSES):
            query_class = ops.bind(pred_encode, concept_tensor[j])
            ## check hamming distance between query class and the original
            hmm_dist = ops.hamming_distance(cls_tensor[j], query_class)
            ## update the dict
            dist_dict[j] = hmm_dist
        ## get the predicted class
        pred_lbl = min(dist_dict, key=dist_dict.get)
        pred_lbl_list.append(pred_lbl)

    return pred_lbl_list

class HDEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HDEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.l_input = nn.Linear(input_size, hidden_size)
        self.l_hidden_1 = nn.Linear(hidden_size, hidden_size)
        self.l_output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.l_input(x)
        x = self.l_hidden_1(x)
        x = self.l_output(x)
        x = self.softmax(x)
        
        bin_x = binarize_outputs(x)

        return x, bin_x


## Test the NN
hd_encoder = HDEncoder(IMG_SIZE, HIDDEN_1, HD_SIZE)
# Set to training mode
hd_encoder.train()
#inputs = torch.rand((1, IMG_SIZE))
#out, bin_out = hd_encoder(inputs)

############################################### TRAIN ###############################################
## cls_tensor holds the sparse binary hd vecs for classes
cls_tensor = make_sparse_bin_tensor(NUM_CLASSES, HD_SIZE)
## concept_tensor stores the class prototypes
concept_tensor = torch.zeros((NUM_CLASSES, HD_SIZE), dtype=torch.int)
## Define the loss function
loss_fn = nn.NLLLoss()
## Define the optimizer
optimizer = optim.SGD(hd_encoder.parameters(), lr=LR)
## Run the network for all the training images
for epoch in range(EPOCH):
    ## predicted encoder list
    pred_encode_list = []
    ## set the optimizer to init
    optimizer.zero_grad()
    for i in range(NUM_IMG_TRAIN):
        img = make_img_tensor(img_batch[i])
        lbl = label[i]
        idx = lbl.numpy()
        ## get the pred scores and corresponding bin hd vector
        pred, pred_encode = hd_encoder(img)
        #pred_encode = pred_encode.detach()
        pred_encode_list.append(pred_encode)
        ## bind the predicted encode with the corresponding label
        ## hd vect
        img_cls_hd_vec = ops.bind(pred_encode, cls_tensor[idx])
        ## bundle the class prototypes with each other within the class
        concept_tensor[idx] = ops.bundle(img_cls_hd_vec, concept_tensor[idx])
    ## Predict the classes from the learnt concepts
    #print(len(pred_encode_list))
    hd_pred_cls = hd_predictor(pred_encode_list, concept_tensor, cls_tensor)
    one_hot_preds = make_one_hot(hd_pred_cls, NUM_CLASSES)
    #print(one_hot_preds.shape)
    #hd_pred_cls = torch.from_numpy(np.array(hd_pred_cls))
    #print(one_hot_preds[0, :])
    #print(hd_pred_cls[0])
    #print(label)
    #print(torch.from_numpy(np.array(hd_pred_cls)))
    
    ## compute the loss
    #for param in hd_encoder.l_input.parameters():
    #    print(param)
    print("")
    loss = loss_fn(one_hot_preds, label)
    print(loss)
    loss = loss + torch.sum(pred * torch.ones(pred.shape, requires_grad=True)) / 1e3
    print(loss.detach().numpy())
    #print(loss)
    ## performa a backward pass
    loss.backward()
    ## update the weight
    optimizer.step()


    

