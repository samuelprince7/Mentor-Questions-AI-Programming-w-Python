import torch

import argparse
import json

from torchvision import datasets, transforms, models
from torch import nn, optim 
import torch.nn.functional as F 
from collections import OrderedDict 
from PIL import Image
import numpy as np
import pandas as pd 

from sp_Artificial_Intelligence import *

from sp_DataTransformations1 import *

with open('ImageClassifier/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

data_dir = 'ImageClassifier/flowers' 

# obtain image data from files and build your data loaders
image_datasets1, dataloaders1 = data_transforms(data_dir, 32)

training_dataloader1 = dataloaders1['train_loader']
val_dataloader1 = dataloaders1['val_loader']

training_datasets1 = image_datasets1['training']

print("The data loaders where successfully created from the flower images data...")

    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

#generate parser
parser = argparse.ArgumentParser(description="Adds command line arguments to Train a new Neural Network using transfer learning in command line app")

#obtain file path to pictures 
parser.add_argument('data_dir', default='ImageClassifier/flowers', help="the file path of images to train the network")

# define the path where you will save the newly created neural network
#should include two checkpoints
# a model_checkpoint AND a detailed_model_checkpoint
parser.add_argument('--save_model_to', default= 'ImageClassifier/saved_model' , help="Creates the path to save the origin model checkpoint")
parser.add_argument('--save_detailed_model_to', default= 'ImageClassifier/saved_detailed_model' , help="Creates the path to save the detailed model checkpoint including the idx mapping")

# select architecture
parser.add_argument('--arch', default="vgg16", help="the arch used to train the model. must be vgg16 or densenet161")

# allow changes to hyperparameters via the application
# this can be learn rate, hidden units, epochs, and batch size
parser.add_argument('--learn_rate', type=float, default="0.001", help ="enter learning rate")
parser.add_argument('--hidden_layer1', type=int, default=4096, help ="enter hidden layer 1")
parser.add_argument('--hidden_layer2', type=int, default=1024, help ="enter hidden layer 2")
parser.add_argument('--epochs', type=int, default=0, help ="enter epochs")
parser.add_argument('--batch_size', type=int, default=32, help ="enter batch size")

# define and ability to utilize the GPU. this one might be kinda tricky given needed parameters
parser.add_argument('--gpu', default=True, action='store_true', help="want to use the GPU? Default is set to False. Must set to True")

args = parser.parse_args()
data_dir = args.data_dir
save_origin_location = args.save_model_to
save_detailed_location = args.save_detailed_model_to
arch = args.arch
learn_rate = args.learn_rate
hidden_layer1 = args.hidden_layer1
hidden_layer2 = args.hidden_layer2
epochs = args.epochs
batch_size = args.batch_size
use_gpu = args.gpu

# author the base model as either vgg16 OR densenet161
# using author_model from sp_Artificial_Intelligence
model = author_model(arch, hidden_layer1, hidden_layer2)
print("your initial model was sucessfully created as...")
print(model)

## ask the program if the GPU is on, AND then assign the 'device' approrpriately as 'cuda' or 'cpu' 
##troubleshooting tests
print("Is the GPU element defined?")
print(args.gpu)
print("Will we try to use the GPU this time?")
print(use_gpu)
if use_gpu == True:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("the GPU is 'ON', IF the device shown below is 'cuda'")
    print("the GPU is 'OFF', IF the device shown below is 'cpu'")
else:
    device = torch.device('cpu')
    print("the GPU is off IF the device shown below is 'cpu'")

print("the device you are running is...")
print(device)

# train the model and print out validation epochs
print(".......")
print("Training the model on the images from the dataloaders now....")
train_the_model(model, training_dataloader1, val_dataloader1, learn_rate, epochs)

print("now that the model has been trained, we will save the origin model AND detailed model, as...")
print(" ..the path you entered after --save_model_to AND --save_detailed_model_to...")
print("...OR if there was an error, the default saved paths are....")
print("...ImageClassifier/saved_model....")
print("...ImageClassifier/saved_detailed_model....")

# because the variable num_of_inputs was defined inside of the xyz function
#we need to also simultaneously define it on this side of the program in trian.py
#so the equivilent variable scope can be applied below in sp_save_model
if arch.lower() == 'vgg16':    
    num_of_inputs = 25088
        
elif arch.lower() == 'densenet161':
    num_of_inputs = 2208  #(carry the same ratios between layers)
else:
    print("Unsupported architecture. Enter vgg16 or densenet161")
    
sp_save_model(model, training_datasets1, num_of_inputs, 102, epochs, learn_rate, arch, hidden_layer1, hidden_layer2, save_origin_location, save_detailed_location)
print("...the origin model and detailed model have been saved...")
# last edit 9:40pm Thur 2/18
                    
                    
