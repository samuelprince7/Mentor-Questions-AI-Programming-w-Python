import torch
from torch import n as nn
from torch import optim as optim
import sp_Artificial_Intelligence
import spDataTransformations

import argparse

#generate parser
parser = argparse.ArgumentParser(description="Train a new Neural Network using transfer learning")

#obtain file path to pictures 
parser.add_argument('data_dir', default='flowers', help="the file path of images to train the network")

# define the path where you will save the newly created neural network
#should include two checkpoints
# a model_checkpoint AND a detailed_model_checkpoint
parser.add_argument('--save_model', default= './' , help="Creates the path to save the origin model checkpoint")
parser.add_argument('--save_detailed_model', default= './' , help="Creates the path to save the detailed model checkpoint with idx")

# select architecture
parser.add_argument('--arch', default="vgg16", help="the arch used to train the model. must be vgg16 or densenet161")

# allow changes to hyperparameters via the application
# this can be learn rate, hidden units, epochs, and batch size
parser.add_argument('--learning_rate', type=float, default="0.001", help ="enter learning rate")
parser.add_argument('--hidden_layer1', type=int, default=4096, help ="enter hidden layer 1")
parser.add_argument('--hidden_layer2', type=int, default=1024, help ="enter hidden layer 2")
parser.add_argument('--epochs', type=int, default=20, help ="enter epochs")
parser.add_argument('--batch_size', type=int, default=32, help ="enter batch size")

# define and ability to utilize the GPU. this one might be kinda tricky given needed parameters
parser.add_argument('--gpu', default=False, action='store_true', help="want to use the GPU? Default is set to False. Must set to True")

# last edit 9:40pm Thur 2/18
