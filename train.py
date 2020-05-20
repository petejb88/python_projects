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
parser.add_argument('--save_dir', default="", help="Location to save the model checkpoint (default: in-place)")
parser.add_argument('--arch', default = 'vgg16_bn', help="Torchvision pretrained model architecture to use (default = vgg16_bn)")
parser.add_argument('--hidden_units', default=[4096,1024], help="Size of the two layers of hidden units (default=[4096,1024])", type=list)
parser.add_argument('--epochs', default=6, type=int, help="Epochs (default=6)")
parser.add_argument('--print_every', default=5, type=int, help="How often to print loss/accuracy statistics (default=5)")
parser.add_argument('--gpu', action="store_true", default=False, help="Use GPU (default=False)")
parser.add_argument('--lr', default=0.001, type=float, help="Learning rate (default=0.001)")
parser.add_argument('--batch_size', default=64, type=int, help="batch size (defautl=64)")
parser.add_argument('--graph_stats', action="store_true", default=False, help="Display training and validation loss plots (default:False)")
# parser.add_argument('--cat_to_name', default='cat_to_name.json', help="Dictionary of categorical class outputs and image names")

args = parser.parse_args()

# with open(args.cat_to_name, 'r') as f:
#    cat_to_name = json.load(f)

from model_functions import *
    
# THE MAIN PROGRAM
def main():
    train_data, valid_data, trainloader, validloader = build_loader(args.data_dir,args.batch_size)
    model, criterion, optimizer = build_model(args.arch,args.lr,args.hidden_units,train_data)    
    model, optimizer, train_losses, valid_losses, accuracy_data = train_model(model,criterion,optimizer,trainloader,validloader,args.epochs,args.print_every,args.gpu)
    if args.graph_stats:
        plot_stats(train_losses,valid_losses)
    create_checkpoint(model,optimizer,args.arch,args.lr,args.epochs,args.save_dir)

if __name__ == "__main__":
    main()
