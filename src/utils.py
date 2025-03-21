import torch
import torch.nn as nn
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
import h5py
from torch import norm

def complex_sep(A):
    real_part = A.real
    imag_part = A.imag
    A_np = np.concatenate((real_part, imag_part), axis=1)
    return A_np


def get_measurement(data_filename):
    """
    Extract measurement samples from .mat file. Separate the real and imaginary parts and computes the mean and std for
    each channel.
    """
    data = loadmat(data_filename)
    H_np = data["H_test"].transpose(3, 2, 0, 1).astype(np.csingle)
    Phi = data["Phi_test"].transpose(3, 2, 0, 1).astype(np.csingle)
    Psi = data["Psi_test"].transpose(3, 2, 0, 1).astype(np.csingle)
    y = data["y_test"].transpose(3, 2, 0, 1).astype(np.csingle)
    h = data["h_test"].transpose(3, 2, 0, 1).astype(np.csingle)

    H_np = complex_sep(H_np)

    mean = np.mean(H_np, axis=(0, 2, 3), keepdims=True).reshape(-1)
    std = np.std(H_np, axis=(0, 2, 3), keepdims=True).reshape(-1)

    return torch.from_numpy(H_np), mean, std, torch.from_numpy(Phi), torch.from_numpy(Psi), torch.from_numpy(y), torch.from_numpy(h)


class MeasDataset(Dataset):
    def __init__(self, H, Phi, Psi, y, h, transform=None):
        self.H = H
        self.Phi = Phi
        self.Psi = Psi
        self.y = y
        self.h = h
        self.transform = transform
        # self.target_transform = target_transform

    def __getitem__(self, idx):
        channel = self.H[idx, :, :, :]
        measurement_matrix = self.Phi[idx, :, :, :]
        noise_projection_matrix = self.Psi[idx, :, :, :]
        measurement = self.y[idx, :, :, :]
        channel_vec = self.h[idx, :, :, :]

        if self.transform:
            channel = self.transform(channel)

        return channel, measurement_matrix, noise_projection_matrix, measurement, channel_vec

    def __len__(self):
        return self.H.shape[0]


class Measurement(Dataset):
    def __init__(self, H, h, Phi, snr, transform=None):
        self.H = H
        self.Phi = Phi
        self.h = h
        self.snr = snr  # in dB
        self.transform = transform
        self.power_sq = norm(torch.einsum('ba, ncaj -> ncbj', Phi, h).reshape(H.shape[0], -1), dim=1)
        self.var = self.power_sq.mean() / (10 ** (snr / 20)) / np.sqrt(h.shape[1] * Phi.shape[0] * 2)
        # self.target_transform = target_transform

    def __getitem__(self, idx):
        H = self.H[idx, :, :, :]
        h = self.h[idx, :, :, :]
        noise = self.var * (torch.randn([h.shape[0], self.Phi.shape[0], 1]) + 1j * torch.randn([h.shape[0], self.Phi.shape[0], 1]) )
        if self.snr == 1000:
            noise = torch.zeros([h.shape[0], self.Phi.shape[0], 1])
        y = torch.einsum('ba, caj -> cbj', self.Phi, h) + noise
        if self.transform:
            channel = self.transform(H)

        return channel, h, y

    def __len__(self):
        return self.H.shape[0]


def get_channel(data_filename):
    """extract the channel matrix stored in .mat, and compute and mean and var for each channel"""
    # read data
    with h5py.File(data_filename, 'r') as file:
        H_np = file['H'][()]
    n, k, t, r = H_np.shape
    H_np = (H_np['real'] + 1j * H_np['imag'])
    h_np = H_np.reshape(n, k, t*r, 1) # vectorize the channel matrix column-wise
    H_np = complex_sep(H_np.transpose(0, 1, 3, 2))  # separate the real and imag parts
    # compute mean and std
    mean = np.mean(H_np, axis=(0, 2, 3), keepdims=True).reshape(-1)
    std = np.std(H_np, axis=(0, 2, 3), keepdims=True).reshape(-1)
    return torch.from_numpy(H_np).to(torch.float32), torch.from_numpy(h_np).to(torch.complex64), mean, std


class ChannelDataset(Dataset):
    def __init__(self, H, h, transform=None):
        self.H = H
        self.h = h
        self.transform = transform
        # self.target_transform = target_transform

    def __getitem__(self, idx):
        channel_mat = self.H[idx, :, :, :]
        channel_vec = self.h[idx, :, :, :]

        if self.transform:
            channel_mat = self.transform(channel_mat)

        return channel_mat, channel_vec

    def __len__(self):
        return self.H.shape[0]


def transform_reverse(Gz, mean, std):
    "Gz shoule be of size (N, C, imag_size, imag_size)"
    C = Gz.shape[1]
    Gz_rev = torch.zeros_like(Gz)
    for c in range(C):
        Gz_rev[:,c, :, :] = Gz[:,c, :, :] * std[c] + mean[c]
    # Gz_rev = Gz * std[:, None, None].from_numpy + mean[:, None, None].from_numpy
    return Gz_rev

# def transform(A, mean, std):
#     "A shoule be of size (N, C, W, H)"
#     # C = A.shape[1]
#     # Gz_rev = torch.zeros_like(Gz)
#     # for c in range(C):
#     #     Gz_rev[:,c, :, :] = Gz[:,c, :, :] - mean[c] /std[c]
#     A_trans = (A - mean[:, None, None]) / std[:, None, None]
#     return A_trans


def vectorize(Gz):
    "Gz shoule be of size (N, C, imag_size, imag_size). vectorize the last 2 dimension"
    Gz_t = Gz.mT
    Gz_vec = Gz_t.reshape(Gz.shape[0],Gz.shape[1],Gz.shape[2]*Gz.shape[3],1)
    return Gz_vec


def to_rgb(fake_imgs_2channel):
    """
    Convert 2-channel images to RGB by adding a zero third channel.

    Args:
    - fake_imgs_2channel (torch.Tensor): Tensor representing the 2-channel images.

    Returns:
    - torch.Tensor: RGB images.
    """
    if fake_imgs_2channel.shape[1] == 2:
        zeros = torch.zeros_like(fake_imgs_2channel[:, :1, :, :])
        rgb_imgs = torch.cat((fake_imgs_2channel, zeros), 1)
    elif fake_imgs_2channel.shape[1] >= 2:
        rgb_imgs = fake_imgs_2channel[:, :3, :, :]
    return rgb_imgs


def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty, gradient_norm.mean()


def get_measurement_matrix(Nt, Nr, Np, Ns, Nrft, Nrfr, seed):
    """generate a measurement matrix of size (Ns*Np, Nt*Nr); analog beamforming bit = 1; digital beamformer =
    identity for all subcarriers; pilots are the same across all subcarriers"""
    torch.manual_seed(seed)
    s = 1 / np.sqrt(Ns) * torch.randn([Np, Ns]) # E{norm(s[i])} = 1
    # print((torch.norm(s, dim=(1,2)) ** 2).mean())
    F = 1 / np.sqrt(Nt/Nrft) * (torch.randint(0, 2, (Np, Nt)) * 2 - 1)
    C = 1 / np.sqrt(Nr/Nrfr) * (torch.randint(0, 2, (Np, Nr)) * 2 - 1)
    Phi = []
    for m in range(Np):
        xmT = (F[m] * torch.repeat_interleave(s[m], Nt // Nrft)).unsqueeze(-1).T
        # only valid for 1 bit quantization
        c1, c2, c3, c4 = C[m].chunk(Nrfr)
        CH = torch.block_diag(c1, c2, c3, c4)
        Phi.append(torch.kron(xmT, CH))
    return torch.cat(Phi, dim=0)


# Phi = get_measurement_matrix(256, 32, 100, 4, 4, 4, seed=417)
# print(Phi.shape)
# def save_checkpoint(state, filename="celeba_wgan_gp.pth.tar"):
#     print("=> Saving checkpoint")
#     torch.save(state, filename)
#
#
# def load_checkpoint(checkpoint, gen, disc):
#     print("=> Loading checkpoint")
#     gen.load_state_dict(checkpoint['gen'])
#     disc.load_state_dict(checkpoint['disc'])
# data_filename = f"data/channel-r32t256-n100.mat"
# H_all, h_all, mean, std = get_channel(data_filename)
