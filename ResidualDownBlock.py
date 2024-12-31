import torch
import torch.nn as nn

import ResidualConvBlock;

# Residual Down Block
class ResidualDownBlock(nn.Module):
  def __init__(self, input_channels, output_channels):
    super(ResidualDownBlock, self).__init__()

    # List of layers for down-sampling
    # And each layer consist of 2 ResidualConvBlock, and MaxPool2d layer
    layer = [
        ResidualConvBlock(input_channels, output_channels),
        ResidualConvBlock(output_channels, output_channels),
        nn.MaxPool2d(2)
    ]

    # Creating a Sequential model using above layers
    self.model = nn.Sequential(*layer)

  def forward(self, x: torch.Tensor):
    # Applying the input x to the model and returning the output
    return self.model(x)