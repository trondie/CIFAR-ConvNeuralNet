import torchvision
from torch import nn
from dataloaders import load_cifar10
from task2 import Trainer, create_plots

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10) # No need to apply softmax,
        # as this is done in nn.CrossEntropyLoss
        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters(): # Unfreeze the last fully-connected
            param.requires_grad = True # layer
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional
            param.requires_grad = True # layers

    def forward(self, x):
        # Upsample to 256x256
        x = nn.functional.interpolate(x, scale_factor=8)
        x = self.model(x)
        return x

batchSize = 32
learningRate = 5e-4
earlyStopCount = 7
epochs = 10
model = Model()
dataloaders = load_cifar10(batchSize)
adam = True
trainedModel = Trainer(
    batchSize,
    learningRate,
    earlyStopCount,
    epochs,
    model,
    dataloaders,
    adam
)
trainedModel.train()
create_plots(trainedModel, "Task 4a")