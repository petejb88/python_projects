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


# load model
def load_checkpoint(checkpoint):
    ''' 
    Loads checkpoint, outputs model with weights from checkpoint
    
    Output:
        - model: trained model
    '''
    checkpoint = torch.load(checkpoint)
    print("Checkpoint loaded")
    arch = checkpoint['pretrained_model']
    
    method = getattr(models,arch)
    if callable(method):
        if (('vgg' in arch) or ('alex' in arch)):
            model = method(pretrained=True)
        else: 
            print("This program is not yet equipped to deal with that architecture.")
    else:
        print("That is not a valid model architecture.")


    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['linear_layers_size'][0])),
        ('relu1', nn.ReLU()),
        ('drop1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(checkpoint['linear_layers_size'][0],checkpoint['linear_layers_size'][1])),
        ('relu2', nn.ReLU()),
        ('drop2', nn.Dropout(p=0.2)),
        ('fc3', nn.Linear(checkpoint['linear_layers_size'][1], checkpoint['output_size'])),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    print("Model created")
    return model


# Process image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''       
    process_transforms = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224), 
                                             transforms.ToTensor(),
                                             transforms.Normalize(
                                                [0.485, 0.456, 0.406], 
                                                [0.229, 0.224, 0.225])])    
    return process_transforms(image)

# Predict image
def predict(image_path,model,topk=1,gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
    Inputs:
        - image_path: path to image
        - model: model to use
        
    Outputs:
        - top_probs: top probabilities
        - top_classes: classes associated to these probabilities
    '''
    
    device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu) else "cpu")
    model = model.to(device)
    
    image = Image.open(image_path).convert('RGB')
    processed_image = process_image(image)[:3,:,:].unsqueeze(0)
    processed_image = processed_image.to(device)
    log_ps = model.forward(processed_image)
    probs = torch.exp(log_ps)
    top_probs, top_indexes = probs.topk(topk)
    # ---- this goes from classes to index, need to go the other direction 
    # top_classes = [model.class_to_idx[str(i)] for i in top_indexes.tolist()[0]]
    top_classes = []
    for i in top_indexes.tolist()[0]:
        classes = [k for (k,v) in model.class_to_idx.items() if v == i]
        top_classes.append(classes[0])
    return top_probs.tolist()[0], top_classes



# display prediction
def graph_predictions(image_path,top_probs, top_classes,category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    top_names = [cat_to_name[str(c)] for c in top_classes]

    y_pos = np.arange(len(top_names))

    from matplotlib.pyplot import figure

    fig, (ax1, ax2) = plt.subplots(nrows=2,figsize=(3,6))

    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('Prediction')
    image = Image.open(image_path)
    ax1.imshow(image)

    ax2.barh(y_pos,top_probs)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_names)
    ax2.set_xlabel('probability')
    
    plt.show(block=True)

# print predictions
def print_predictions(top_probs,top_classes,category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    names = [cat_to_name[str(c)] for c in top_classes]

    for prob,name in zip(top_probs,names):
        print("Object: {}, Probability: {}".format(name,prob))
