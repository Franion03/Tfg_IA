import torch
import torch.nn as nn
from .utils import load_state_dict_from_url
from typing import Any


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        """
         Initializes the AlexNet class. It is used to train and test the model. If you don't want to run this yourself you can call : meth : ` super ` instead.
         
         Args:
         	 num_classes: Number of classes to predict. Default is 1000
        """
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
         Forward pass of the neural network. Takes a batch of features and averages them across the batch to produce a batch of predictions
         
         Args:
         	 x: input features of shape [ batch_size num_features ]
         
         Returns: 
         	 batch of predictions of shape [ batch_size num_features ] - > [ batch_size num
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    """
     Factory for AlexNet model architecture. This is a class method to create a : class : ` ~arxiv. paper. AlexNet ` object with pre - trained weights
     
     Args:
     	 pretrained: whether to return a model pre - trained on ImageNet
     	 progress: whether to display a progress bar of the download to stderr
     	 kwargs: keyword arguments to pass to : class : ` ~arxiv. paper. AlexNet `
     
     Returns: 
     	 an instance of : class : ` ~arxiv. paper. AlexNet ` seealso :: : func : ` ~gensim. models. alexnet
    """
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    # Load the state dictionary from the model_urls alexnet.
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model