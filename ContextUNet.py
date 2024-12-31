import torch
import torch.nn as nn

import EmbedBlock
import ResidualUpBlock
import ResidualDownBlock
import ResidualConvBlock



# Context UNET
class ContextUNet(nn.Module):
  def __init__(self, input_channels, n_features=256, n_context_features=10, height=28):
    super(ContextUNet, self).__init__()

    # Initializing the given values (channels, features, classes, height)
    self.height = height
    self.n_features = n_features
    self.input_channels = input_channels
    self.n_context_features = n_context_features

    # Initializing the initial Convolutional Layer
    self.init_conv = ResidualConvBlock(self.input_channels, self.n_features,
                                        is_res_block=True)

    # Initializing the down_sampling path of the U-Net with 2 levels
    # UNET Down Blocks
    self.down1 = ResidualDownBlock(self.n_features, self.n_features)      # down1: [10, 256, 8, 8]
    self.down2 = ResidualDownBlock(self.n_features, self.n_features * 2)  # down2: [10, 256, 4, 4]

    # Original: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
    self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())

    # Embed the timestep and context labels with a one-layer fully connected neural network
    self.time_embed1 = EmbedBlock(1, n_features * 2)
    self.time_embed2 = EmbedBlock(1, n_features * 1)
    self.context_embed1 = EmbedBlock(n_context_features, n_features * 2)
    self.context_embed2 = EmbedBlock(n_context_features, n_features * 1)

    # Initializing the up-sampling path of the U-Net with 3 levels
    self.up0 = nn.Sequential(
        nn.ConvTranspose2d(n_features * 2, n_features * 2, self.height // 4, self.height // 4),
        nn.GroupNorm(8, n_features * 2),
        nn.ReLU(),
    )

    self.up1 = ResidualUpBlock(n_features * 4, n_features)
    self.up2 = ResidualUpBlock(n_features * 2, n_features)

    # Initialize the final convolutional layers to map to the same number of channels as the input image
    self.out = nn.Sequential(
        # reduce number of feature maps: (input_channels, out_channels, kernel_size, stride=1, padding=1 ??? they put zero here in conmment and 1 in below conv2d layerhow zero find it or may by they are wrong...)
        nn.Conv2d(n_features * 2, n_features, 3, 1, 1), # confirm it... # [10, 256, 4, 4] => [10, 256, 8, 8]
        # normalizing...
        nn.GroupNorm(8, n_features),
        nn.ReLU(),
        # reduce number of feature maps again: (input_channels, out_channels, kernel_size, stride=1, padding=1)
        nn.Conv2d(n_features, self.input_channels, 3, 1, 1) # map the same number of channels as input
    )

  def forward(self, x, t, context=None):
    # x: (batch, n_features, h, w)    = input image
    # t: (batch, n_context_features)  =  time step
    # context: (batch, n_classes)     = context label

    # here context_mask says which samples to block the context on

    # passing the input image through the initial convolutional layer
    x = self.init_conv(x)
    # passing the result through the down-sampling path
    down1 = self.down1(x)     # [10, 256, 8, 8]
    down2 = self.down2(down1) # [10, 256, 4, 4]

    # converting the features by maps to a vector and apply an activation
    hiddenvec = self.to_vec(down2)

    # mask out context if context_mask == 1 ??? find out is it none or one (1)
    if context is None:
      context = torch.zeros(x.shape[0], self.n_context_features).to(x)
    # else:
    #   context = context

    # embed context and timestep
    context_embed1 = self.context_embed1(context).view(-1, self.n_features * 2, 1, 1)
    time_embed1 = self.time_embed1(t).view(-1, self.n_features * 2, 1, 1)
    context_embed2 = self.context_embed2(context).view(-1, self.n_features, 1, 1)
    time_embed2 = self.time_embed2(t).view(-1, self.n_features, 1, 1)


    up1 = self.up0(hiddenvec)
    up2 = self.up1(context_embed1 * up1 + time_embed1, down2) # add and multiply embeddings
    up3 = self.up2(context_embed2 * up2 + time_embed2, down1)
    out = self.out(torch.cat((up3, x), 1))

    return out