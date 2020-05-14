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

import argparse


parser = argparse.ArgumentParser(description="Train an classifier using a pre-trained model.")
parser.add_argument('data_dir', help="Location of data, with 'train' and 'valid' subfolders, and images partitioned into further subfolders.")
parser.add_argument('--save_dir', default="/", help="Location to save the model checkpoint (default: '/')")
parser.add_argument('--arch', default = 'vgg16_bn', help="Torchvision pretrained model architecture to use (default = vgg16_bn)")
parser.add_argument('--hidden_units', default=[4096,1024], help="Size of the two layers of hidden units (default=[4096,1024])", type=list)
parser.add_argument('--epochs', default=6, type=int, help="Epochs (default=6)")
parser.add_argument('--print_every', default=5, type=int, help="How often to print loss/accuracy statistics (default=5)")
parser.add_argument('--gpu', action="store_true", default=False, help="Use GPU (default=False)")
parser.add_argument('--lr', default=0.001, type=float, help="Learning rate (default=0.001)")
# parser.add_argument('--graph_stats', action="store_true", default=False)
# parser.add_argument('--cat_to_name', default='cat_to_name.json', help="Dictionary of categorical class outputs and image names")

args = parser.parse_args()

# with open(args.cat_to_name, 'r') as f:
#    cat_to_name = json.load(f)



# BUILD LOADER
def build_loader():
    ''' Creates dataloaders:
    
    Input: 
        - data_dir: data directory
        
    Output:
        - train_data: training data
        - valid_data: validating data
        - trainloader: training data loader
        - validloader: validating data loader
    '''
    data_dir = args.data_dir
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

    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
    
    return(train_data, valid_data, trainloader, validloader)

# BUILD MODEL
def build_model(train_data):
    ''' Build the model based on the specified pretrained model
    
    Input:
        - arch: architecture
        
    Output:
        - model: the model
        - classifier: the new part of the model to be trained
        - optimizer: how we are detemining loss
    '''
    # get architecture
    arch = args.arch
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
    hidden_units = args.hidden_units        
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
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)

    return model, criterion, optimizer


# TRAINING THE MODEL
def train_model(model,criterion,optimizer,trainloader,validloader):    
    ''' Train the model
    
    Input:
        - model: the model to be trained
        - criterion, optimizer: how to determine effectiveness of model
        - trainloader, validloader: image batches        
        
    Output:
        - model: model with better weights
        - train_losses, valid_losses: the losses stats for each time stats were printed
        - accuracy_data: accuracy (on validation data) for each time stats were printed
    '''
    
    epochs = args.epochs
    steps = 0
    print_every = args.print_every
       
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.gpu) else "cpu")
    model.to(device)
    
    train_losses, valid_losses, accuracy_data = [], [], []
    
    for e in range(0,epochs):
        print("Epoch {} starting...".format(e+1))
        running_loss = 0
    
        for image_batch, label_batch in trainloader:
            steps += 1
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
        
            logps = model.forward(image_batch)
            loss = criterion(logps, label_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()        
            
            # stats per print_every steps
            if steps % print_every == 0:
                train_losses.append(running_loss/len(trainloader))
                # validation
                accuracy = 0
                valid_loss = 0
                with torch.no_grad():
                    model.eval()
            
                    for image_batch, label_batch in validloader:
                        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
                
                        valid_logps = model.forward(image_batch)
                        valid_loss += criterion(valid_logps, label_batch)
                        valid_losses.append(valid_loss/len(validloader))
                                    
                        # accuracy computation
                        valid_ps = torch.exp(valid_logps)
                        top_ps, top_labels = valid_ps.topk(1,dim=1)
                        equals = top_labels == label_batch.view(*top_labels.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))
             
                # print stats
                print(f'Epoch {int(e)+1}')
                print(f'Running Loss: {running_loss / print_every}')
                print(f'Validating Accuracy: {accuracy / len(validloader)}')
                accuracy_data.append(accuracy / len(validloader))
                print(f'Validating Loss: {valid_loss / len(validloader)}')
            
                running_loss = 0
                model.train()
                    
    return model, optimizer, train_losses, valid_losses, accuracy_data


# plot stats
def plot_stats(train_losses, valid_losses):
    ''' Plot stats
    
    Inputs:
        - train_losses, valid_losses: list of training/validation losses
    
    Output: 
        - creates a plot
    '''
    
    valid_losses_plot = [valid_losses[n] for n in   range(1,len(valid_losses),len(valid_losses)//len(train_losses))]

    plt.plot(range(0,len(train_losses)),train_losses,label="train losses")
    plt.plot(range(0,len(train_losses)),valid_losses_plot, label="validation losses")
    plt.legend()        


# save checkpoint
def create_checkpoint(model,optimizer):
    
    for i,module in enumerate(model.classifier):
        if type(module) == torch.nn.modules.linear.Linear:
            classifier_input = module.in_features
            break
    
    checkpoint = {'pretrained_model' : args.arch,
                  'input_size' : classifier_input,   
                  'linear_layers_size' : [model.classifier[n].out_features for n,each in enumerate(model.classifier) if type(each) == torch.nn.modules.linear.Linear],
                  'processing_layers' : ['relu','drop'],
                  'output_size' : 102,
                  'output_layer' : 'LogSoftmax',
                  'criterion' : 'nn.NLLLoss()',
                  'optimizer' : 'Adam',
                  'lr' : args.lr,
                  'epochs' : args.epochs,
                  'optimizer_state' : optimizer.state_dict(),
                  'state_dict' : model.state_dict(),
                  'classifier_dict' : model.classifier.state_dict(),
                  'class_to_idx' : model.class_to_idx,
                 }

    torch.save(checkpoint, args.save_dir+'checkpoint.pth')
    
    
# THE MAIN PROGRAM
def main():
    train_data, valid_data, trainloader, validloader = build_loader()
    print("Data loaded")
    model, criterion, optimizer = build_model(train_data)    
    print("Model created")
    model, optimizer, train_losses, valid_losses, accuracy_data =       train_model(model,criterion,optimizer,trainloader,validloader)
    print("Model trained!")
    print("Final validation accuracy: {}".format(accuracy_data[-1]))
    create_checkpoint(model,optimizer)

if __name__ == "__main__":
    main()
