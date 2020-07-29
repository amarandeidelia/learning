import argparse

import torch

from utils import load_data
from model import get_model, train_model, save_model

parser = argparse.ArgumentParser(add_help=True, description='Train a new network on a data set')
parser.add_argument('data_dir', help='The directory for the images which will be used to train the model.')
parser.add_argument('--save_dir', help='The directory to save the model')
parser.add_argument('--arch', default='vgg16', help='The architecture to choose')
parser.add_argument('--learning_rate', type=int, default=0.01, dest='lr', help='The learning rate')
parser.add_argument('--hidden_units', type=int, default=1024, help='The unit of the hidden layer')
parser.add_argument('--epochs', type=int, default=3, help='The number of epochs for the training')
parser.add_argument('--gpu', action='store_true', default=False, help='Switch gpu mode on')

args = parser.parse_args()

if __name__ == '__main__':
    train_dataset, valid_dataset = load_data(data_dir=args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)
    output_size = len(train_dataset.class_to_idx)

    model = get_model(base_model=args.arch, output_size=output_size, hidden_layer=args.hidden_units, dropout=0.5)
    device = 'cuda' if args.gpu else 'cpu'
    train_model(model=model, device=device, lr=args.lr, epochs=args.epochs, train_loader=train_loader,
                valid_loader=valid_loader)
    save_model(base_model=args.arch, model=model, train_dataset=train_dataset, output_size=output_size,
               hidden_layer=args.hidden_units, output_dir=args.save_dir)
