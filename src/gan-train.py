"""
Training of DCWGAN-GP

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-01: Initial coding
* 2022-12-20: Small revision of code, checked that it works with latest PyTorch version
"""
import numpy as np
import argparse
import os
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary
# from src import utils
import utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import Discriminator, Generator, initialize_weights

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--sample_ckpt", type=int, default=5, help="sampling rate for saving models")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--betas", type=float, default=(0.5, 0.99), help="adam: decay of momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=(16, 32, 256), help="size of each image (C,H,W)")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
opt, unknown = parser.parse_known_args()
print(opt)

os.makedirs("ckpt/gan_r32t256k8", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# initialize gen, disc, optimizors
gen = Generator(opt.img_size, opt.latent_dim).to(device)
critic = Discriminator(opt.img_size).to(device)
summary(critic, (1, 16, 32, 256))
summary(gen, (1, 100))
initialize_weights(gen)
initialize_weights(critic)
gen.train()
critic.train()
opt_gen = optim.Adam(gen.parameters(), lr=opt.lr, betas=opt.betas)
opt_critic = optim.Adam(critic.parameters(), lr=opt.lr, betas=opt.betas)

LAMBDA_GP = 10

# load channel dataset
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

# for tensorboard plotting
fixed_noise = torch.randn(32, opt.latent_dim).to(device)
writer_real = SummaryWriter(f"logs/gan_r32t256_e{opt.n_epochs}b{opt.batch_size}/real")
writer_fake = SummaryWriter(f"logs/gan_r32t256_e{opt.n_epochs}b{opt.batch_size}/fake")
writer_loss = SummaryWriter(f"logs/gan_r32t256_e{opt.n_epochs}b{opt.batch_size}/loss")
step = 0

batch_done = 0
for epoch in range(opt.n_epochs+1):
    pbar = tqdm(total=H_all.shape[0] // opt.batch_size, desc=f'Epoch [{epoch+1}/{opt.n_epochs+1}]')
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        
        cur_batch_size = real.shape[0]
        noise = torch.randn(cur_batch_size, opt.latent_dim).to(device)
        fake = gen(noise)
        
        if (batch_idx + 1) % opt.n_critic != 0:
            # Train Critic: max E[critic(real)] - E[critic(fake)]
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp, grad_norm = utils.gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()
            batch_done += 1
            pbar.update(1)
            pbar.set_postfix({'Loss D': loss_critic.item(),
                              'loss G': -torch.mean(critic_fake).item(),
                              'Grad norm': grad_norm.item(),
                              })
        if (batch_idx + 1) % opt.n_critic == 0:
            # Train Generator: max E[critic(gen_fake)]
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            batch_done += 1
            pbar.update(1)
            pbar.set_postfix({'Loss D': loss_critic.item(),
                              'loss G': loss_gen.item(),
                              'Grad norm': grad_norm.item(),
                              })
        # Print losses occasionally and print to tensorboard
        if batch_done % 50 == 0 and batch_idx > 0:
            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(utils.to_rgb(real[:32]), normalize=False)
                img_grid_fake = torchvision.utils.make_grid(utils.to_rgb(fake[:32]), normalize=False)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                writer_loss.add_scalar("loss D", loss_critic, batch_done)
                writer_loss.add_scalar("loss G", loss_gen, batch_done)
                writer_loss.add_scalar("D(x)", critic(real).mean(), batch_done)
                writer_loss.add_scalar("D(G(z))", critic(gen(noise)).mean(), batch_done)
                writer_loss.add_scalar("Grad norm", grad_norm, batch_done)

            step += 1
    pbar.close()
    if epoch % (opt.sample_ckpt) == 0:
        # save ckpt every 20 epoch
        torch.save({
            'generator_state_dict': gen.state_dict(),
            'discriminator_state_dict': critic.state_dict(),
            'opt': opt
        }, f"ckpt/gan_r32t256k8/e{epoch}b{opt.batch_size}.pth.tar")