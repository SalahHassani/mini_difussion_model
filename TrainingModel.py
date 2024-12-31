# importing necessary libraries...
import os
import torch
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader

import ContextUNet
from CustomFunctionsAndClasses import *

# hyperparameters

# diffusion hyperparameters
timesteps = 500
beta1 = 1e-4
beta2 = 0.02

# network hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') # torch.device("cuda")
n_features = 64 # 64 hidden dimension feature
n_context_features = 5 # 5 classes OR context vector of size 5
height = 16 # 16x16 image
save_dir = './weights/'

# training hyperparameters
batch_size = 100
n_epoch = 32
learning_rate = 1e-3 # eta

# construct DDPM noise schedule
beta_at_time_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
alpha_at_time_t = 1.0 - beta_at_time_t
alpha_hat_at_time_t = torch.cumsum(alpha_at_time_t.log(), dim=0).exp()
alpha_hat_at_time_t[0] = 1

# construct model
nn_model = ContextUNet(input_channels=3, n_features=n_features, n_context_features=n_context_features, height=height).to(device)


# load dataset and construct optimizer
dataset = CustomDataset("./data/sprites_1788_16x16.npy", "./data/sprite_labels_nc_1788_16x16.npy", transform, False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate)
# we can apply AdamW which better and faster

# helper function: perturbs an image to a specified noise level
def perturb_input(x, t, noise):
    return alpha_hat_at_time_t.sqrt()[t, None, None, None] * x + (1 - alpha_hat_at_time_t[t, None, None, None]) * noise


def train_mini_diffusion_model():
    nn_model.train()

    for epoch in range(n_epoch):
        print(f'epoch: {epoch}')

    # linearly decay learning rate
    optimizer.param_groups[0]['lr'] = learning_rate * (1 - epoch / n_epoch)

    # iterate over dataset
    pbar = tqdm(dataloader, mininterval=2)
    for x, _ in pbar:
        # x: images
        optimizer.zero_grad()
        x = x.to(device)

        # pertube data
        noise = torch.randn_like(x)
        t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
        x_pert = perturb_input(x, t, noise)

        # use model or network to recover noise
        pred_noise = nn_model(x_pert, t / timesteps)

        # loss in mean squared error between the predicted and true noise
        loss = F.mse_loss(pred_noise, noise)
        loss.backward()

        # update weights
        optimizer.step()

    # save model perdiocally, for every 4 epochs and final epoch
    if epoch % 4 == 0 or epoch == int(n_epoch - 1):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(nn_model.state_dict(), save_dir + f"context_model_{epoch}.pth")
        print('saved model at ' + save_dir + f"context_model_{epoch}.pth")