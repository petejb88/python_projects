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
parser.add_argument('--topk', default=5, type=int, help="Return the top K most likely classes (default=5)")
parser.add_argument('--category_names', default = 'cat_to_name.json', help="Dictionary mapping categorical class outputs to actual names (default: 'cat_to_name.json')")
parser.add_argument('--gpu', action="store_true", default=False, help="Use GPU (default=False)")
parser.add_argument('--graph_pred', action="store_true", default=False, help="Graph predictions (default=False)")

args = parser.parse_args()

    
# main program
def main():
    model = load_checkpoint(args.checkpoint)
    print("Checkpoint Loaded!")
    print("Model created!")
    top_probs, top_classes = predict(args.image_path,model,args.topk,args.gpu)
    if args.graph_pred:
        graph_predictions(args.image_path,top_probs,top_classes,args.category_names)
    else:
        print_predictions(top_probs,top_classes,args.category_names)
    

if __name__ == "__main__":
    main()
