import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, note_per_beat=4, n_beat=8, device="cpu"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.note_per_beat = note_per_beat
        self.n_beat = n_beat
        self.device = device

        self.beta = self.prepare_noise_schedule()
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_hat = self.alpha_hat.to(device)
        

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_partitions(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 2*self.note_per_beat*self.n_beat)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x)
                alpha = self.alpha[t][:, None]
                alpha_hat = self.alpha_hat[t][:, None]
                beta = self.beta[t][:, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x_note, x_duration = torch.chunk(x, 2, 1)
        x_note *= NOTE_NORMALIZATION_A
        x_note += NOTE_NORMALIZATION_B
        x_duration *= DURATION_NORMALIZATION
        x = torch.cat((x_note, x_duration), 1)
        return x

def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet(device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(note_per_beat=args.note_per_beat, n_beat=args.n_beat, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    for epoch in tqdm(range(args.epochs)):
        logging.info(f"Starting epoch {epoch}:")
        for i, partitions in enumerate(dataloader):
            # partitions = partitions.to(device)
            t = diffusion.sample_timesteps(1)
            x_t, noise = diffusion.noise_partitions(partitions, t)
            predicted_noise = model(x_t)[0]
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.add_scalar("MSE", mse(noise, x_t[0]).item(), global_step=epoch * l + i)

        # sampled_partitions = diffusion.sample(model, n=partitions.shape[0])
        # save_partitions(sampled_partitions, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), f"models/{args.run_name}/ckpt.pt")


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "what_dense_model"
    args.epochs = 2000
    args.batch_size = 12
    args.note_per_beat = 4
    args.n_beat = 8
    args.dataset_path = r"musicnet/train_labels"
    args.device = torch.device("cpu")
    args.lr = 3e-4
    
    train(args)

if __name__ == '__main__':
    launch()
    # device = torch.device("cpu")
    # model = UNet().to(device)
    # model.load_state_dict(torch.load("./models/first_model/ckpt.pt"))
    # diffusion = Diffusion(device=device)
    # x = diffusion.sample(model, 1)
    # print(x.shape)
    # play_song_with_array(x.to("cpu").numpy()[0])
