import torch
import torch.nn as nn

import ResidualConvBlock

# Residual Up Block
class ResidualUpBlock(nn.Module):
  def __init__(self, input_channels, output_channels):
    super(ResidualUpBlock, self).__init__()

    # Creating a list of layers for the up-sampling block
    # The Block consist of a ConvTranspose2d layer for upsampling,
    # followed by two ResidualConvBlock layers
    layer = [
        # (input_channels, output_channels, kernel_size=2, stride=2)
        nn.ConvTranspose2d(input_channels, output_channels, 2, 2),
        ResidualConvBlock(output_channels, output_channels),
        ResidualConvBlock(output_channels, output_channels)
    ]

    # Creating a Sequential model using the above layers
    self.model = nn.Sequential(*layer)

  def forward(self, x: torch.Tensor, skip: torch.Tensor):
    # Concatenating the input tensor x with the skip connection tensor alone the dimension
    x = torch.cat((x, skip), 1)

    # passing the concatenated tensor through the sequential model and return the output
    return self.model(x)