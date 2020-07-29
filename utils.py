import torch
from torchvision import datasets, transforms
from PIL import Image


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    train_transforms = transforms.Compose(
        [transforms.RandomRotation(30),
         transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    return train_dataset, valid_dataset


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    im = Image.open(image_path)

    transformer = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return transformer(im)


def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Implement the code to predict the class from an image file
    image = process_image(image_path)
    image.to(device)
    image.unsqueeze_(0)
    with torch.no_grad():
        output = model.forward(image)
    ps = torch.exp(output)
    props, idx = ps.data.topk(topk, dim=1)
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[i] for i in idx[0].tolist()]
    return props[0].tolist(), classes
