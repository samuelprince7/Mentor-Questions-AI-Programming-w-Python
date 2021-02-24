import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms, models
from torch import nn, optim 
import torch.nn.functional as F 
from collections import OrderedDict 
from PIL import Image
import numpy as np
import pandas as pd 

import ProcessImageNimshow

import json
import Load_Json_File

import argparse 

import sp_DataTransformations1 

import sp_Artificial_Intelligence

#generate a parser to take in specifications from the command line application
parser = argparse.ArgumentParser(description ="Load a pretrained network & test for inference on a single image")

#below are the 'arguments' for the user to imput via the command line
#description of the argument is listed in the help section
# this command line application was designed with simplicity in mind
# leveraging the complexity of the sp_Artificial_Intelligence program
# to test newly trained neural networks

parser.add_argument('data_dir', help="path to the image to preform testing on")
parser.add_argument('checkpoint_pth', help="path to a newly trained neural network MODEL checkpoint as pth file.")
parser.add_argument('detailed_checkpoint_pth', help="path to a newly trained neural network DETAILED MODEL checkpoint as pth file.")
parser.add_argument('--top_probs', default=1, type=int, help="number of most probable classes you wish to preview. this shows a number of flowers the pretrained model has identified as probable flower names")

parser.add_argument('--spec_labels_pth', default = 'cat_to_name.json',help="json file path to load flower names")
parser.add_argument('--gpu', default=False, action='store_true', help="set to True to use your GPU")

# enable input from the command line
args = parser.parse_args()
data_dir = args.data_dir
checkpoint = args.checkpoint_pth
detailed_checkpoint = args.detailed_checkpoint_pth
top_probs = args.top_probs
names = args.spec_labels_pth
utilize_gpu = args.gpu

# rebuild the model
rebuilt_model = sp_Artificial_Intelligence.simple_load_rebuild_model(checkpoint) 
rebuilt_detailed_model = sp_Artificial_Intelligence.simple_load_rebuilt_model(detailed_checkpoint)

#get json data that holds the names of flowers to an idx mapping 
flower_names = Load_Json_File.load_json_file(names)
#run predict with the rebuilt model and this should
# output the top_k results of the rebuilt model in the terminal
probs, classes = sp_Artificial_Intelligence.predict(data_dir, rebuilt_model, utilize_gpu, top_probs)
print(probs)
print(classes)

#flower_names should go in sanity checker along with the detailed checkpoint (create detailed checkpoit above)
# see if you can use the sanity checker and if you can't you'll have to 
# rewrite a version of the sanity checker that only prints the flower names & probabilities 
# instead of plotting it using matplotlib

sp_Artificial_Intelligence.sp_sanity_checker(flower_names, data_dir, rebuilt_model, rebuilt_detailed_model)
#789
