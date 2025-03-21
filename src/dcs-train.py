# generate measurement matrix while training

import numpy as np
import argparse
import os
import sys
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import utils
from torch import tanh
from torch.linalg import norm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import Generator, initialize_weights
from torchinfo import summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
    parser.add_argument("--sample_ckpt", type=int, default=5, help="sampling rate for saving models")
    parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--lr_est", type=float, default=0.1, help="adam: learning rate")
    parser.add_argument("--betas", type=float, default=(0.9, 0.99), help="adam: decay of momentum of gradient")
    parser.add_argument("--betas_est", type=float, default=(0.9, 0.99), help="adam: decay of momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=(16, 32, 256), help="size of each image (C,H,W)")
    parser.add_argument("--gd_step", type=int, default=10, help="gradient descent steps")
    parser.add_argument("--n_p", type=int, default=100, help="number of pilots")  
    opt = parser.parse_args()
    opt, unknown = parser.parse_known_args()
    print(opt)

    os.makedirs("ckpt/dcs_r32t256k8", exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    gen = Generator(opt.img_size, opt.latent_dim).to(device)
    initialize_weights(gen)
    opt_gen = optim.Adam(gen.parameters(), lr=opt.lr, betas=opt.betas)
    summary(gen, (1, opt.latent_dim))

    H_all, h_all, mean, std = torch.load('./data/channel-r32t256k8-n5000.pt')
    std_norm = np.linalg.norm(std)
    transform = transforms.Compose([
        transforms.Normalize(mean, std)  # Normalizing for all channels.
    ])
    dataloader = torch.utils.data.DataLoader(
        utils.ChannelDataset(H_all, h_all, transform=transform),
        batch_size=opt.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )

    Phi = utils.get_measurement_matrix(256, 32, opt.n_p, 4, 4, 4, 417).to(device).to(torch.complex64)

    def phi_mat(a):
        return torch.einsum('ba, ncaj -> ncbj', Phi, a)

    def real2complex(A):
        # convert from real tensor of size [N,2K,W,H] to a complex tensor [N,K,W,H]
        C = A.shape[1] // 2
        A_complex = A[:, :C, :, :] + 1j * A[:, C:, :, :]
        return A_complex
    
    def z2h(z, reverse=True):
        Gz = gen(z)
        Gz_rev = utils.transform_reverse(Gz, mean, std)
        if reverse is True:
            gz = utils.vectorize(Gz_rev)
        else:
            gz = utils.vectorize(Gz)
        # recover complex channels
        return real2complex(gz)
    
    def loss_inner(m_i, Phi, z):
        """"m_i, Phi, z -> [Batch, ]"""
        n_batch = m_i.shape[0]
        gz = z2h(z, reverse=True)
        err = (norm((m_i - phi_mat(gz)).reshape(n_batch,-1),dim=1) / std_norm ) ** 2
        # err = err + opt.loss_reg * (norm(z) ** 2)
        return err
    
    def loss_outer(err_recon, Phi, x1, x2, x3):
        loss_G = err_recon.mean()
        n_batch = err_recon.shape[0]
    
        def get_rip_loss(img1, img2):
            m1 = (phi_mat(img1)).reshape(n_batch, -1)
            m2 = (phi_mat(img2)).reshape(n_batch, -1)
            img_diff_norm = norm((img1 - img2).reshape(n_batch, -1), dim=1)
            m_diff_norm = norm(m1 - m2, dim=1)
            return (img_diff_norm / m_diff_norm - np.sqrt(Phi.shape[1] / Phi.shape[0])) ** 2 * (
                    Phi.shape[0] / Phi.shape[1])
        # measurement loss (triplet loss)
        r1 = get_rip_loss(x1, x2)
        r2 = get_rip_loss(x1, x3)
        r3 = get_rip_loss(x2, x3)
        loss_F = ((r1 + r2 + r3) / 3.0).mean()
    
        return loss_G, loss_F
    
    def nmse_compute(z, h_bch):
        n_batch = z.shape[0]
        gz = z2h(z, reverse=True)
        err = torch.linalg.norm((h_bch - gz).reshape(n_batch, -1), dim=1) ** 2
        h_norm = torch.linalg.norm(h_bch.reshape(n_batch, -1), dim=1) ** 2
        return err / h_norm
    

    # ----------
    #  Training
    # ----------
    # torch.autograd.set_detect_anomaly(True) # for debug
    # batches_done = 0
    fixed_noise = torch.randn(32, opt.latent_dim).to(device)
    fixed_noise /= fixed_noise.norm(dim=1, keepdim=True)
    writer_real = SummaryWriter(f"logs/dcs_r32t256k8_e{opt.n_epochs}b{opt.batch_size}gd{opt.gd_step}np{opt.n_p}dl{opt.latent_dim}/real")
    writer_fake = SummaryWriter(f"logs/dcs_r32t256k8_e{opt.n_epochs}b{opt.batch_size}gd{opt.gd_step}np{opt.n_p}dl{opt.latent_dim}/fake")
    writer_loss = SummaryWriter(f"logs/dcs_r32t256k8_e{opt.n_epochs}b{opt.batch_size}gd{opt.gd_step}np{opt.n_p}dl{opt.latent_dim}/loss")
    batches_done = 0

    def sample_and_opt(m, h):
        z = torch.randn([opt.batch_size, opt.latent_dim], requires_grad=True, device=device)
        opt_z = optim.Adam([z], lr=opt.lr_est, betas=opt.betas_est)
        gen_img_initial = z2h(z.clone())
        for _ in range(opt.gd_step):
            opt_z.zero_grad()
            err_recon = loss_inner(m, Phi, z)
            err_recon.backward(torch.ones_like(err_recon))
            opt_z.step()
            with torch.no_grad():
                z /= z.norm(dim=1, keepdim=True)
                pass
        err_recon = loss_inner(m, Phi, z)
        gen_img_final = z2h(z.clone())
        nmse = nmse_compute(z, h)
        return nmse, gen_img_initial, gen_img_final

    for epoch in range(opt.n_epochs+1):
        # tic()
        pbar = tqdm(total=H_all.shape[0] // opt.batch_size, desc=f'Epoch [{epoch+1}/{opt.n_epochs+1}]')
        for batch, (H, h) in enumerate(dataloader):
            H, h = H.to(device), h.to(device)

            m = phi_mat(h)  # to avoid repmat Phi to save memory
            nmse, gen_img_initial, gen_img_final = sample_and_opt(m, h)
            loss_G, loss_F = loss_outer(nmse, Phi, h, gen_img_initial, gen_img_final)

            loss_meta = loss_G + loss_F

            opt_gen.zero_grad()
            loss_meta.backward()
            opt_gen.step()

            batches_done += 1
            pbar.update(1)
            pbar.set_postfix({'Loss mata': loss_meta.item(),
                              'nmse': nmse.mean().item(),
                              })

            # Track loss
            with torch.no_grad():
                writer_loss.add_scalar("loss meta", loss_meta.item(), batches_done),
                writer_loss.add_scalar("nmse", nmse.mean().item(), batches_done),

            # Track generator output
            if batches_done % 50 == 0:
                with torch.no_grad():
                    fake = gen(fixed_noise)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(utils.to_rgb(tanh(H[:32])), normalize=False)
                    img_grid_fake = torchvision.utils.make_grid(utils.to_rgb(tanh(fake[:32])), normalize=False)

                    writer_real.add_image("Real", img_grid_real, global_step=batches_done)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=batches_done)
        pbar.close()

        if epoch % (opt.sample_ckpt) == 0:
            torch.save({
                'generator_state_dict': gen.state_dict(),
                'opt': opt
            }, f"ckpt/dcs_r32t256k8/e{epoch}b{opt.batch_size}gd{opt.gd_step}np{opt.n_p}dl{opt.latent_dim}.pth.tar")
            print(f"epoch {epoch}: model saved ")


if __name__ == "__main__":
    main()
