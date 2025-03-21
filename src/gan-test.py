import numpy as np
import argparse
import torch
import torch.optim as optim
import torchvision
from model import Generator
import utils
from torch import norm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from utils import AdamManual

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_est", type=float, default=1e-1, help="adam: learning rate in online stage")
    parser.add_argument("--betas_est", type=float, default=(0.9, 0.99),
                        help="adam: decay of momentum of gradient in online stage")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=(16, 32, 256), help="size of each image (C,H,W)")
    # parser.add_argument("--np_dtype", default=np.csingle, help="numpy data type")
    parser.add_argument("--gd_step", type=int, default=100, help="gradient descent steps")
    parser.add_argument("--n_test", type=int, default=100, help="number of test data")
    parser.add_argument("--n_restart", type=int, default=10, help="number of restart")
    # parser.add_argument("--loss_reg", type=float, default=1e-3, help="regularization coeff of inner loss")
    # parser.add_argument("--log", default=False, help="log or not")
    # parser.add_argument("--log_file", default="logs/test", help="log file name")
    opt, unknown = parser.parse_known_args()
    print(opt)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the trained GAN model, and get information about the training data
# load the trained GAN model, and get information about the training data
    ckpt = torch.load(f"./ckpt/gan_r32t256k8/e55b100.pth.tar", map_location=device)
    # ckpt = torch.load(f"ckpt_s/dcs_e1000b500.pth.tar", map_location=device)
    gen = Generator(opt.img_size, opt.latent_dim).to(device)
    gen.load_state_dict(ckpt['generator_state_dict'])
    gen.eval()

    # training dataset
    _, _, mean, std = torch.load('./data/channel-r32t256k8-n5000.pt')
    std_norm = np.linalg.norm(std)
    
    # testing dataset
    H_all, h_all, _, _ = torch.load('./data/channel-r32t256k8-n1000.pt')
    # std_norm = np.linalg.norm(std)
    
    Phi = utils.get_measurement_matrix(256, 32, 100, 4, 4, 4, 417).to(torch.complex64)

    SNR_vec = [
        -20,
        -15,
        -10,
        -5,
        0,
        5,
        10,
        15,
        # 20,
        # 25,
        # 30
    ]
    nmse_avg = torch.zeros((len(SNR_vec)))
    nmse_avg_db = torch.zeros((len(SNR_vec)))


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
        err = (norm((m_i - phi_mat(gz)).reshape(n_batch, -1), dim=1) / std_norm) ** 2
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
        err = norm((h_bch - gz).reshape(n_batch, -1), dim=1) ** 2
        h_norm = norm(h_bch.reshape(n_batch, -1), dim=1) ** 2
        return err / h_norm


    for snr_idx, snr in enumerate(SNR_vec):
        # writer = SummaryWriter(f"{opt.log_file}/snr{snr}")
        # print(f"SNR: {snr}")
        # data_filename = f"dataset/Test_Rx16Tx16_SNR_{snr}_1000_varying_angles.mat"
        # H_all, _, _, Phi_all, Psi_all, y_all, h_all = utils.get_measurement(data_filename)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean, std)  # Normalizing for all channels.
        ])
        TestDataloader = torch.utils.data.DataLoader(
            utils.Measurement(H_all[:opt.n_test], h_all[:opt.n_test], Phi.to("cpu"), snr, transform=transform),
            batch_size=opt.n_test,
            shuffle=True,
            pin_memory=True,
            num_workers=0
        )
        # nmse = torch.zeros((len(TestDataloader)))
        restart, step = 0, 0
        pbar = tqdm(total=opt.n_restart * opt.gd_step, desc=f'SNR {snr}: [{snr_idx+1}/{len(SNR_vec)}]')
        for batch, (H, h, y) in enumerate(TestDataloader):
            H, y, h, Phi = H.to(device), y.to(device), h.to(device), Phi.to(device)
    
            # verify snr
            signal_power = norm(phi_mat(h).reshape(1000, -1), dim=1) ** 2
            noise_power = norm(y.reshape(1000, -1) - phi_mat(h).reshape(1000, -1), dim=1) ** 2
            snr_db = (10*torch.log10(signal_power/noise_power)).mean()
            # print(snr_db)
            cur_bch = H.shape[0]
    
            # initialize best z
            z_best = torch.randn([cur_bch, opt.latent_dim], requires_grad=False, device=device)
            err_recon_best = loss_inner(y, Phi, z_best).detach()
            for restart in range(opt.n_restart):
                z = torch.randn([cur_bch, opt.latent_dim], requires_grad=True, device=device)
                optimizer_z = optim.Adam([z], lr=opt.lr_est, betas=opt.betas_est)
                for step in range(opt.gd_step):
                    optimizer_z.zero_grad()
                    err_recon = loss_inner(y, Phi, z)
                    err_recon.backward(torch.ones_like(err_recon))
                    optimizer_z.step()
                    # with torch.no_grad():
                    #     z /= z.norm(dim=1, keepdim=True)
                    #     err_recon = loss_inner(y, Phi, z)
                    #     z_best = z.detach().clone()
                    #     pbar.update(1)
                    with torch.no_grad():
                        err_recon = loss_inner(y, Phi, z)
                        idx = torch.where(err_recon < err_recon_best)
                        err_recon_best[idx] = err_recon[idx].detach().clone()
                        z_best[idx] = z[idx].detach().clone()
                        pbar.update(1)
            nmse = nmse_compute(z_best, h).detach()
        pbar.close()
        pass
        nmse_avg[snr_idx] = nmse.mean()
        nmse_avg_db[snr_idx] = 10 * torch.log10(nmse_avg[snr_idx])
        print(f"nmse: {nmse_avg[snr_idx]:.2f} = {nmse_avg_db[snr_idx]:.2f} dB")
        # with torch.no_grad():
        #     writer.add_scalar(f"nmse_avg", nmse_avg[snr_idx], snr)
        #     writer.add_scalar(f"nmse_db_avg", nmse_avg_db[snr_idx], snr)
    print(nmse_avg)
    print(nmse_avg_db)
    # nmse_filename = f"nmse_db_dcwgangp.csv"
    # np.savetxt(nmse_filename, nmse_avg_db.numpy(), delimiter=',')
    # nmse_filename = f"nmse_db_dcwgangp.csv"
    # np.savetxt(nmse_filename, nmse_avg_db.numpy(), delimiter=',')

