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

parser = argparse.ArgumentParser(description="Predict image name from a model checkpoint.")
parser.add_argument('image_path', help="Path to image")
parser.add_argument('checkpoint', help='Checkpoint for model')
parser.add_argument('--topk', default=5, help="Return the top K most likely classes (default=5)")
parser.add_argument('--category_names', default = 'cat_to_name.json', help="Dictionary mapping categorical class outputs to actual names (default: 'cat_to_name.json')")
parser.add_argument('--gpu', action="store_true", default=False, help="Use GPU (default=False)")

args = parser.parse_args()



# load model
def load_checkpoint():
    ''' 
    Loads checkpoint, outputs model with weights from checkpoint
    
    Output:
        - model: trained model
    '''
    checkpoint = torch.load(args.checkpoint)
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
def predict(image_path, model):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
    Inputs:
        - image_path: path to image
        - model: model to use
        
    Outputs:
        - top_probs: top probabilities
        - top_classes: classes associated to these probabilities
    '''
    topk=args.topk
    
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.gpu) else "cpu")
    model = model.to(device)
    
    image = Image.open(image_path)
    processed_image = process_image(image).unsqueeze(0)
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
def graph_predictions(top_probs, top_classes):
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    names = [cat_to_name[str(c)] for c in top_classes]

    y_pos = np.arange(len(names))

    from matplotlib.pyplot import figure

    fig, (ax1, ax2) = plt.subplots(nrows=2,figsize=(3,6))

    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('Prediction')
    ax1.imshow(image_path)

    ax2.barh(y_pos,top_probs)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_names)
    ax2.set_xlabel('probability')
    
    plt.show(block=True)

# print predictions
def print_predictions(top_probs,top_classes):
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    names = [cat_to_name[str(c)] for c in top_classes]

    for prob,name in zip(top_probs,names):
        print("Object: {}, Probability: {}".format(name,prob))
    
# main program
def main():
    model = load_checkpoint()
    print("Checkpoint Loaded!")
    print("Model created!")
    top_probs, top_classes = predict(args.image_path,model)
    print_predictions(top_probs,top_classes)
    

if __name__ == "__main__":
    main()
