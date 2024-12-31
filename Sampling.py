from TrainingModel import *

# helper function: removes the predicted noise (but adds some noise back in ot avoid collapse)
def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = beta_at_time_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - alpha_at_time_t[t]) / (1 - alpha_hat_at_time_t[t]).sqrt())) / alpha_at_time_t[t].sqrt()
    return mean + noise


# sample using standard algorithm
@torch.no_grad
def sample_ddpm(n_sample, save_rate=20):
    # x_T ~ N(0, 1), sample initial noise
    sample = torch.randn((n_sample, 3, height, height)).to(device)

    # array to keep track of generated steps for plotting
    intermediate = []
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(sample) if i > 1 else 0

        eps = nn_model(sample, t)
        sample = denoise_add_noise(sample, i, eps, z)

        if i % save_rate == 0 or i == timesteps or i < 8:
            intermediate.append(sample.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return sample, intermediate
#