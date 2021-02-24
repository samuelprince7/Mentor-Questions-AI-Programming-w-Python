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

import sp_DataTransformations1 

image_datasets1, dataloaders1 = data_transforms(data_dir, 32)

training_dataloader1 = dataloaders1['train_loader']
val_dataloader1 = dataloaders1['val_loader']

training_datasets1 = image_datasets1['training']

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
def author_model(arch, hidden_layer1, hidden_layer2):
    '''Manufactures a new pretrained model using VGG16 or (Densenet161) & returns it
    
    Inputs: 
    arch - the pytorch architecture in all lower case
    hidden_layer1 = count of elements in the 1st hidden layer
    hidden_layer2 = count of elements in the 2nd hidden layer
    
    Output:
    model - newly authored pretrained model 
    
    '''
    
    #load in the pretrained network(vgg16 or densenet161)
    # define an untrained feed-forward network as a classifier
    
    print("Building model now....")
    
    #Load model
    if arch.lower() == 'vgg16':
        model = models.vgg16(pretrained=True)
        num_of_inputs = 25088
        
    elif arch.lower() == 'densenet161':
        model = models.densenet161(pretrained=True)
        num_of_inputs = 2208  #(carry the same ratios between layers)
        
    else:
        print("Unsupported architecture. Enter vgg16 or densenet161")
        # return 0 
              
    # freeze parameters so there is no backpropigation happening          
    for param in model.parameters():
        param.requires_grad = False 
    
    classifier = nn.Sequential(nn.Linear(num_of_inputs, hidden_layer1),
                          nn.ReLU(),
                          nn.Dropout(p=0.5), #0.5 is good, returns half for standardization
                          nn.Linear(hidden_layer1, hidden_layer2),
                          nn.ReLU(),
                          nn.Dropout(p=0.5), #0.5 is good
                          nn.Linear(hidden_layer2, 102), 
                          nn.LogSoftmax(dim=1))

    model.classifier = classifier    
    print("This model has been created as...\n")
    print(model)
    return model 
          
def train_the_model(model, training_dataloaders, val_dataloaders, learn_rate, epochs, use_gpu): 
    #TODO: still need to change the variables for the training_dataloaders & val_dataloaders    
    '''
    Trains the model. Prints output loss and accuracy
    
    Inputs:
    model - the newly authored model to now train
    training_dataloaders - transformed data for training
    val_dataloaders - transformed data for validation
    learn_rate - the learning rate
    epochs - how many epochs the training will run
    use_gpu - true or false
    
    Outputs:
    Prints the training and validation losses and accuracies
    Trains the model
    
    '''
    
    #use the GPU if it is avaialbe
          #as the user indicated above
    if use_gpu:
          device = torch.device('cuda')
    else:
          device = torch.device('cpu')
          
    # allow the model to be trained on the device (either cpu or gpu) 
    model.to(device)
    
    #define the loss     
    criterion = nn.NLLLoss()
          
    #establish learning rate      
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate) #should almost always be 0.001  
          
    # defining the primary variables used in our training program
    epochs = epochs #(should be 20-25 for a good model)
    steps = 0
    running_loss = 0
    print_every = 40      #(assusmes 20-ish epochs)
          
    # build the 'training loop'
    # this loop says, "this is how we learn from the data "
    for epoch in range(epochs):
        
    # we use the images formated into tensors from dataloaders['train_loader']
        for images, labels in training_dataloaders: # dataloaders['train_loader']:
        #we take a step in our learning journey 
            steps += 1
            # move the tensor data over to the GPU so it can process everything even faster
            images, labels = images.to(device), labels.to(device)
            # zero out gradients
            optimizer.zero_grad()
            #obtain the log probabilities
            logps = model(images)
            #logps = model.forward(images)
            # now w/the log probabilities we can get the loss from the criterion in the labels
            loss = criterion(logps, labels)
            # do a backwards pass, or backpropigation so it can learn from it's own mistakes
            loss.backward()
            # now it learns one step forward, so we pass that as a forward step in the learning model
            # it's kind of like we are telling it how to think and learn from it's previous mistake
            #found in the backprop 
            optimizer.step()
            # finally we have to increment the running loss, so we keep track of of our total 'training loss'
            # and thus the program learns from itself continually 
            # until the end of it's cycle 
            running_loss += loss.item()
        
            #####VALIDATION LOOP 
        
            if steps % print_every ==0:
                #turn our model into evaluation inference mode which turns off dropout
                # this allows using the network to make predictions
                model.eval() # turn off drop-out
                validation_loss = 0
                accuracy = 0
                # get the images and label tensors fro the testing set in dataloaders['test_loader']
                for images, labels in val_dataloaders:  # dataloaders['val_loader']:
                    #transfer tensors over to the GPU
                    images, labels = images.to(device), labels.to(device)
                    logps = model(images)
                    loss = criterion(logps, labels)
                    validation_loss += loss.item()
                    # calculate the accuracy
                    ps = torch.exp(logps) # get probabilities
                    top_ps, top_class = ps.topk(1, dim=1) # returns the 1st largest values in the probabilities 
                    # check for equality by creating an equality tensor
                    equality = top_class == labels.view(*top_class.shape)
                    #calculate accuaracy from the equality 
                    accuracy += torch.mean(equality.type(torch.FloatTensor))
        
                    #we want to keep track of our epochs so we'll use the f string format
                print(f"Epoch {epoch+1}/{epochs}..."
                    f"Train loss: {running_loss/print_every:.3f}..."
                    f"Validation loss: {validation_loss/len(val_dataloaders):.3f}..."  #len(dataloaders['val_loader'])
                    f"Validation accuracy: {accuracy/len(val_dataloaders):.3f}")  #len(dataloaders['val_loader'])
              
                running_loss = 0
                # Turn drop-out back on      
                model.train()     
          
def sp_save_model(model, training_datasets, input_size, output_size, epochs, learning_rate, arch, hidden_layer1, hidden_layer2):      
    '''extremely important output of the model captured including
    optimizer_state, class_to_idx, state_dict '''
    #saves the model checkpoint
    torch.save(model, 'model_checkpoint1.pth')
    
    model.class_to_idx = training_datasets.class_to_idx  # image_datasets['training'].class_to_idx
    
    detailed_checkpoint = {'input_size': input_size,
             'output_size': output_size,
             'epochs': epochs,
             'learning_rate': learning_rate,
             'optimizer_state': optimizer.state_dict(),
             'arch': arch,
             'class_to_idx': model.class_to_idx,
             'hidden_layer1': hidden_layer1,
             'hidden_layer2': hidden_layer2,          
             'state_dict': model.state_dict()}
    
    #saves the detailed checkpoint
    torch.save(detailed_checkpoint, 'detailed_checkpoint1.pth')
          

def simple_load_rebuild_model(path):
    trained_model = torch.load(path)
    return trained_model      
          
#now rebuild PARTICULAR models from checkpoints (rebuild a model checkpoint AND a detailed checkpoint)
p_rebuilt_model = simple_load_rebuild_model('model_checkpoint1.pth')
p_rebuilt_detailed_model = simple_load_rebuild_model('detailed_checkpoint1.pth')
       
          
#please note that predict takes in ONLY the p_rebuilt_model
          # and NOT p_rebuilt_detailed_model
          
def predict(image_path, rebuilt_model, gpu_is_on, topk=top_probs):    #def predict(image_path, rebuilt_model, gpu_is_on, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
     You need to pass it p_rebuilt_model, NOT p_rebuilt_detailed_model...
     The detailed model is only used in the dictionaries of sanity_checker in order to 
     map out the flower names accordingly to the idx
    '''
    device = torch.device("cuda" if utilize_gpu else "cpu") # gpu_is_on torch.cuda.is_available() 
    
    pic = process_image(image_path) #creates format of what was previously called 'test_image' from ee
    pic = pic.to(device)              
    pic = pic.unsqueeze(0)
    product = rebuilt_model.forward(pic)
    product = torch.exp(product)
    probs, classes = product.topk(topk, largest =True, sorted=True)
    probabilities = probs.data
    classes = classes.data
    probs_list = probabilities.tolist()
    classes_list = classes.tolist() 
    probs = probs_list[0]
    classes = classes_list[0]
    return probs, classes
          

#the sanity checker takes in both 
# p_rebuilt_model   as it's model_checkpoint AND
# p_rebuilt_detailed_model  as it's detailed_checkpoint 
          
          
# for below enter p_rebuilt_model1, p_rebuilt_model          
def sp_sanity_checker(cat_to_name, image_path, rebuilt_model, rebuilt_detailed_model):
    #you may need to include the entire predict funciton above this
    #use the model_checkpoint
    probs, classes = predict(image_path, rebuilt_model)
    
    #map key=classes to value=probabilities 
    p_c_dict = dict()
    for idx in range(0, len(probs)):
        p_c_dict[classes[idx]] = probs[idx]
        
    #build a dictionary including the class_to_idx from the model    
    dict_origin_class2idx = rebuilt_detailed_model['class_to_idx']
    dict_origin_class2idx = dict((value, key) for key,value in dict_origin_class2idx.items())

    #now map the flower IDnumber to the probability 
    new_order_prob_class_dict = dict()
    for key, values in dict_origin_class2idx.items():
        for k, v in p_c_dict.items():
            if str(key) == str(k):
                new_order_prob_class_dict[values]=v

    #now map the flower name to the probability 
    flower_prob_list = []
    for key, value in new_order_prob_class_dict.items():
        flower_prob_dictionary = dict()
        for k, v in cat_to_name.items():
            if key == k:
                flower_prob_dictionary['name'] = v
                flower_prob_dictionary['prob'] = value
        flower_prob_list.append(flower_prob_dictionary)
        
    #now create a simple pandas dataframe from flower_prob_list
    #so it can be easily mapped and plotted to matplotlib
    dataframe = pd.DataFrame(flower_prob_list)
    #sort from lowest to highest probability 
    dataframe = dataframe.sort_values(by=['prob'])
    
    #now plot the pic of the flower with it's name
    # and the approriated probabilities for the top 5 flowers
    plt.figure(1)
    plt.subplot(121)
    plt.title(str(dataframe.iloc[-1]['name']))
    imshow((process_image(image_path)), plt, str(dataframe.iloc[-1]['name']))

    plt.figure(2)
    plt.subplot(222)

    plt.barh(range(len(dataframe['name'])),dataframe['prob'])
    plt.yticks(range(len(dataframe['name'])),dataframe['name'])
    plt.show()
          
# last update   rebuilt_model, rebuilt_detailed_model   at 8:53pm  
