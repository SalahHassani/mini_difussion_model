import torch.nn as nn

# Embedding Block
class EmbedBlock(nn.Module):
  def __init__(self, input_dim, embed_dim):
      super(EmbedBlock, self).__init__()

      # NOTE
      # It defines a generic one layer feed-forward neural network for embedding
      # input data of dimensionality input_dim to and embedding space of dimensionality embed_dim

      self.input_dim = input_dim

      # defining the layers for the network
      layers = [
          nn.Linear(input_dim, embed_dim),
          nn.GELU(),
          nn.Linear(embed_dim, embed_dim)
      ]

      # creating a sequential model using the defined layers
      self.model = nn.Sequential(*layers)

  def forward(self, x):

      # flatten the given input tensor x
      x = x.view(-1, self.input_dim)
      # applying the input tensor x to the model and returning the output
      return self.model(x)