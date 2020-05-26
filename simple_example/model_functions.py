# the functions we actually use in train.py
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import numpy as np

import json
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt



# BUILD LOADER
def build_loader(data_dir,batchsize):
    ''' Creates dataloaders:
    
    Input: 
        - data_dir: data directory
        
    Output:
        - train_data: training data
        - valid_data: validating data
        - trainloader: training data loader
        - validloader: validating data loader
    '''
    train_dir = data_dir+"/train/"
    valid_dir = data_dir+"/valid/"

    train_transforms = transforms.Compose([transforms.RandomRotation(90),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                               [0.485, 0.456, 0.406], 
                                               [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                               [0.485, 0.456, 0.406], 
                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size = batchsize, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = batchsize)

    print("Data loaded")
    return(train_data, valid_data, trainloader, validloader)

# BUILD MODEL
def build_model(arch,learningrate,hidden_units,train_data):
    ''' Build the model based on the specified pretrained model
    
    Input:
        - arch: architecture
        - hidden_units: list containing the number of nodes in each of the two hidden layers
        - learningrate: learning rate for the model
        - train_data: path to training data
    Output:
        - model: the model
        - classifier: the new part of the model to be trained
        - optimizer: how we are detemining loss
    '''
    # determine if architecture is compatible
    method = getattr(models,arch)    
    if callable(method):
        if (('vgg' in arch) or ('alex' in arch)):
            model = method(pretrained=True)
        else: 
            print("This program is not yet equipped to deal with that architecture.")
    else:
        print("That is not a valid model architecture.")

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # replace classifier        
    for i,module in enumerate(model.classifier):
        if type(module) == torch.nn.modules.linear.Linear:
            classifier_input = module.in_features
            break
    hidden_units = hidden_units        
    classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(classifier_input, hidden_units[0])),
            ('relu1', nn.ReLU()),
            ('drop1', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(hidden_units[0],hidden_units[1])),
            ('relu2', nn.ReLU()),
            ('drop2', nn.Dropout(p=0.2)),
            ('fc3', nn.Linear(hidden_units[1], 102)),
            ('output', nn.LogSoftmax(dim=1))
         ]))    
    model.classifier = classifier
       
    model.class_to_idx = train_data.class_to_idx
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learningrate)

    print("Model created")
    return model, criterion, optimizer


# TRAINING THE MODEL
def train_model(model,criterion,optimizer,trainloader,validloader,epochs,print_every,gpu,print_steps=False):    
    ''' Train the model
    
    Input:
        - model: the model to be trained
        - criterion, optimizer: how to determine effectiveness of model
        - trainloader, validloader: image batches        
        - epochs: number of epochs to train
        - print_every: how often to print/compute validation stats
        - gpu: T/F whether to use the gpu
        
    Output:
        - model: model with better weights
        - train_losses, valid_losses: the losses stats for each time stats were printed
        - accuracy_data: accuracy (on validation data) for each time stats were printed
    '''
    
    device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu) else "cpu")
    model.to(device)
    
    train_losses, valid_losses, accuracy_data = [], [], []
    
    for e in range(0,epochs):
        print("Epoch {} starting...".format(e+1))
        running_loss = 0
        train_count = 0
        steps = 0  
    
        for image_batch, label_batch in trainloader:
            if print_steps:
                print(steps)
            steps += 1
            train_count += len(image_batch)
            
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
        
            logps = model.forward(image_batch)
            loss = criterion(logps, label_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()        
            
            # stats per print_every steps
            if steps % print_every == 0:
                # validation
                accuracy = 0
                valid_loss = 0
                valid_count = 0
                with torch.no_grad():
                    model.eval()
            
                    for image_batch, label_batch in validloader:
                        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
                
                        valid_count += len(image_batch)
                        
                        valid_logps = model.forward(image_batch)
                        valid_loss += criterion(valid_logps, label_batch)
                                    
                        # accuracy computation
                        valid_ps = torch.exp(valid_logps)
                        top_ps, top_labels = valid_ps.topk(1,dim=1)
                        equals = (top_labels == label_batch.view(*top_labels.shape))
                        accuracy += torch.sum(equals.type(torch.FloatTensor))
                
                # update stat lists
                accuracy_data.append(accuracy / valid_count)
                valid_losses.append(valid_loss / valid_count)
                train_losses.append(running_loss / train_count) # (len(trainloader)*print_every) )
                       
                # print stats
                print(f'Epoch {int(e)+1}')
                print(f'Running Loss: {running_loss / train_count}') # (len(trainloader)*print_every)}')
                print(f'Validating Accuracy: {accuracy / valid_count}')
                print(f'Validating Loss: {valid_loss / valid_count}')
                running_loss = 0
                train_count = 0
                model.train()

    print("Model trained!")
    print("Final validation accuracy: {}".format(accuracy_data[-1]))
    return model, optimizer, train_losses, valid_losses, accuracy_data


# plot stats
def plot_stats(train_losses, valid_losses):
    ''' Plot stats
    
    Inputs:
        - train_losses, valid_losses: list of training/validation losses
    
    Output: 
        - creates a plot
    '''
    
    valid_losses_plot = [valid_losses[n] for n in  range(0,len(valid_losses),len(valid_losses)//len(train_losses))]

    plt.plot(range(0,len(train_losses)),train_losses,label="train losses")
    plt.plot(range(0,len(train_losses)),valid_losses_plot, label="validation losses")
    plt.legend()
    plt.show()


# save checkpoint
def create_checkpoint(model,optimizer,arch,lr,epochs,save_dir):
    
    for i,module in enumerate(model.classifier):
        if type(module) == torch.nn.modules.linear.Linear:
            classifier_input = module.in_features
            break
    
    checkpoint = {'pretrained_model' : arch,
                  'input_size' : classifier_input,   
                  'linear_layers_size' : [model.classifier[n].out_features for n,each in enumerate(model.classifier) if type(each) == torch.nn.modules.linear.Linear],
                  'processing_layers' : ['relu','drop'],
                  'output_size' : 102,
                  'output_layer' : 'LogSoftmax',
                  'criterion' : 'nn.NLLLoss()',
                  'optimizer' : 'Adam',
                  'lr' : lr,
                  'epochs' : epochs,
                  'optimizer_state' : optimizer.state_dict(),
                  'state_dict' : model.state_dict(),
                  'classifier_dict' : model.classifier.state_dict(),
                  'class_to_idx' : model.class_to_idx,
                 }

    torch.save(checkpoint, save_dir+'checkpoint.pth')
    print("Checkpoint created at {}checkpoint.pth".format(save_dir))
