import torch
import torch.nn as nn


# Residual Convolutional Block
class ResidualConvBlock(nn.Module):
  def __init__(self, input_channels, output_channels, is_res_block=False) -> None:
    super().__init__()

    # Checking if input and output channels are same
    self.same_channels = input_channels == output_channels

    # Setting the Flag weather to use Residual connection or not
    self.is_res_block = is_res_block

    # First Convolutional Layer
    self.conv1 = nn.Sequential(
        # (input, output, 3x3 kernel with stride 1 and padding 1)
        nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
        # Batch Normalization
        nn.BatchNorm2d(output_channels),
        # GELU Activation Function
        nn.GELU()
    )

    # Second Convolution Layer
    self.conv2 = nn.Sequential(
        # (input, output, 3x3 kernel with stride 1 and padding 1)
        nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
        # Batch Normalization
        nn.BatchNorm2d(output_channels),
        # GELU Activation Function
        nn.GELU()
    )


  def forward(self, x: torch.Tensor) -> torch.Tensor:

    # Here we use the Flag, weather to use Residual Connection
    if self.is_res_block:
      # Applying both convolution layer one after anther
      x1 = self.conv1(x)
      x2 = self.conv2(x1)

      # above we safe weather both input and output are same or not
      # we use it here if yes then we add residual connection
      if self.same_channels:
        out = x + x2
      else:
        # If not we apply 1x1 convolutional layer to match the
        # dimensions before adding residual connection
        shortcut = nn.Conv2d(x.shape[1], x2.shape[1], kernel_size=1,
                              stride=1, padding=0).to(x.device)

        out = shortcut(x) + x2

      # Normalize output tensor, here 1.414 is constant to normalze
      # the tensor which authors choose
      return out / 1.414

    else:
      # Incase when not using residual connection we apply the convolutional
      # layers and return the output
      x = self.conv1(x)
      x = self.conv2(x)
      return x

  def get_output_channels(self):
    return self.conv2[0].out_channels

  def set_output_channels(self, output_channels):
    self.conv1[0].output_channels = output_channels
    self.conv2[0].in_channels = output_channels
    self.conv2[0].out_channels = output_channels
