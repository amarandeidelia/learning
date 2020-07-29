import argparse
import json

import torch

from utils import process_image, predict
from model import load_checkpoint

parser = argparse.ArgumentParser(add_help=True, description='Predict flower name from an image')
parser.add_argument('input', help='The path to the image')
parser.add_argument('checkpoint', help='The checkpoint from which to load the model')
parser.add_argument('--top_k', type=int, default=3, help='The number of how many of the top classes will be returned')
parser.add_argument('--category_names', default='cat_to_name.json', help='The mapping file of categories to real names')
parser.add_argument('--gpu', action='store_true', default=False, help='Switch gpu mode on')

args = parser.parse_args()

if __name__ == '__main__':
    device = 'cuda' if args.gpu else 'cpu'
    model = load_checkpoint(args.checkpoint)

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    probs, classes = predict(image_path=args.input, model=model, topk=args.top_k, device=device)
    class_names = [cat_to_name[str(c)] for c in classes]

    for p, c in zip(probs, class_names):
        print(f'Class: {c}, Probability: {p}')
