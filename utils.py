import os
import torch
import pandas as pd
from matplotlib import pyplot as plt
from settings import model_setting

NOTE_NORMALIZATION_A = 80
NOTE_NORMALIZATION_B = 20
DURATION_NORMALIZATION = 8
STARTING_TIME_NORMALIZATION = 100

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_partitions(partitions, path, **kwargs):
    """NOT DONE""" # TODO
    pass

def get_data(args):
    data = []
    
    for partition_name in os.listdir(args.dataset_path):
        with open(f"{args.dataset_path}/{partition_name}", 'r') as file:
            partition = pd.read_csv(file)
            
            starting_times = torch.tensor(partition["start_beat"][:32].values, dtype=torch.float, device=args.device) / STARTING_TIME_NORMALIZATION
            notes = (torch.tensor(partition["note"][:32].values, dtype=torch.float, device=args.device) - NOTE_NORMALIZATION_B) / NOTE_NORMALIZATION_A
            durations = torch.tensor(partition["end_beat"][:32].values, dtype=torch.float, device=args.device) / DURATION_NORMALIZATION
            
            data.append(torch.cat((notes, starting_times, durations)))
    return data

def post_process_sequence_batch(batch_tuple):
    
    input_sequences, output_sequences, lengths = batch_tuple
    
    splitted_input_sequence_batch = input_sequences.split(split_size=1)
    splitted_output_sequence_batch = output_sequences.split(split_size=1)
    splitted_lengths_batch = lengths.split(split_size=1)

    training_data_tuples = zip(splitted_input_sequence_batch,
                               splitted_output_sequence_batch,
                               splitted_lengths_batch)

    training_data_tuples_sorted = sorted(training_data_tuples,
                                         key=lambda p: int(p[2]),
                                         reverse=True)

    splitted_input_sequence_batch, splitted_output_sequence_batch, splitted_lengths_batch = zip(*training_data_tuples_sorted)

    input_sequence_batch_sorted = torch.cat(splitted_input_sequence_batch)
    output_sequence_batch_sorted = torch.cat(splitted_output_sequence_batch)
    lengths_batch_sorted = torch.cat(splitted_lengths_batch)
    
    input_sequence_batch_sorted = input_sequence_batch_sorted[:, -lengths_batch_sorted[0, 0]:, :]
    output_sequence_batch_sorted = output_sequence_batch_sorted[:, -lengths_batch_sorted[0, 0]:, :]
    
    input_sequence_batch_transposed = input_sequence_batch_sorted.transpose(0, 1)
    
    lengths_batch_sorted_list = list(lengths_batch_sorted)
    lengths_batch_sorted_list = map(lambda x: int(x), lengths_batch_sorted_list)
    
    return input_sequence_batch_transposed, output_sequence_batch_sorted, list(lengths_batch_sorted_list)

def interpolate(model, n_interpolations, sequence_length, sos=None, device = "cuda" if torch.cuda.is_available() else "cpu"):

  # # Get input.

  z1 = torch.randn((1,1,model_setting["latent_size"])).to(device)
  z2 = torch.randn((1,1,model_setting["latent_size"])).to(device)

  tone1 = model.inference(sequence_length, z1, sos)
  tone2 = model.inference(sequence_length , z2, sos)

  alpha_s = torch.linspace(0,1,n_interpolations)

  interpolations = torch.stack([alpha*z1 + (1-alpha)*z2  for alpha in alpha_s])


  samples = [model.inference(sequence_length ,z, sos) for z in interpolations]


  samples = torch.stack(samples)

  return samples, tone1, tone2


def plot_elbo(losses, mode):
    elbo_loss = list(map(lambda x: x[0].item(), losses))
    kl_loss = list(map(lambda x: x[1], losses))
    recon_loss = list(map(lambda x: x[2], losses))

    losses = {"elbo": elbo_loss, "kl": kl_loss, "recon": recon_loss}
    print(losses)
    for key in losses.keys():
        plt.plot(losses.get(key), label=key + "_" + mode)

    plt.legend()
    plt.show()