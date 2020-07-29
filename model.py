from collections import OrderedDict

import torch
from torch import nn, optim
from torchvision import models

model_input_size = {
    'vgg16': 25088,
    'densenet161': 1024
}


def get_model(base_model, output_size, hidden_layer, dropout=0.5):
    if base_model == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif base_model == 'densenet161':
        model = models.densenet161(pretrained=True)
    else:
        raise(f'Model {base_model} is not supported')

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(model_input_size[base_model], hidden_layer)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(dropout)),
        ('fc2', nn.Linear(hidden_layer, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    return model


def validation(model, valid_loader, criterion, device):
    test_loss = 0
    accuracy = 0
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output).data
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.cuda.FloatTensor()).mean()

    return test_loss, accuracy


def train_model(model, device, lr, epochs, train_loader, valid_loader):
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    steps = 0
    running_loss = 0
    print_every = 40
    for e in range(epochs):
        model.train()
        for images, labels in train_loader:
            steps += 1

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, valid_loader, criterion, device)

                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss / len(valid_loader)),
                      "Test Accuracy: {:.3f}".format(accuracy / len(valid_loader)))

                running_loss = 0

                # Make sure training is back on
                model.train()


def save_model(base_model, model, train_dataset, output_size, hidden_layer, output_dir=None):
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {
        'base_model': base_model,
        'output_size': output_size,
        'hidden_layer': hidden_layer,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    if output_dir:
        file_path = output_dir + '/' + 'checkpoint.pth'
    else:
        file_path = 'checkpoint.pth'
    torch.save(checkpoint, file_path)


def load_checkpoint(file_path):
    checkpoint = torch.load(file_path)
    model = get_model(
        base_model=checkpoint['base_model'],
        output_size=checkpoint['output_size'],
        hidden_layer=checkpoint['hidden_layer']
    )
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model
