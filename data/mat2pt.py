from src import utils
import torch

data_filename = f"data/channel-r32t256k8-n1000.mat"
H_all, h_all, mean, std = utils.get_channel(data_filename)
torch.save((H_all, h_all, mean, std), './data/channel-r32t256k8-n1000.pt')