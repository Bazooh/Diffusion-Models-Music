
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import post_process_sequence_batch
from settings import global_setting, training_setting, model_setting
from dataset import PianoGenerationDataset
from tqdm import tqdm
from utils import plot_elbo
from midi.midi_utils import midiwrite

class UNet(nn.Module):
    def __init__(self, note_per_beat=4, n_beat=8, device="cpu"):
        super().__init__()
        
        n_out = 3*note_per_beat*n_beat
        
        self.final_norm = nn.Conv1d(n_out, n_out, 1, device=device)
        

    def forward(self, x: torch.Tensor):
        
        return self.final_norm(x)

class LSTM_MUSIC_VAE(torch.nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, latent_size, num_layers=1, device="cpu"):
        super(LSTM_MUSIC_VAE, self).__init__()

        # Variables
        self.num_layers = num_layers
        self.lstm_factor = num_layers
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.device = device

        # Embedding
        self.embed = torch.nn.Linear(in_features= self.vocab_size , out_features=self.embed_size, device=device)

        # Encoder Part
        self.encoder_lstm = torch.nn.LSTM(input_size= self.embed_size,hidden_size= self.hidden_size, batch_first=True, num_layers= self.num_layers)
        self.mean = torch.nn.Linear(in_features= self.hidden_size * self.lstm_factor, out_features= self.latent_size, device=device)
        self.log_variance = torch.nn.Linear(in_features= self.hidden_size * self.lstm_factor, out_features= self.latent_size, device=device)

        # Decoder Part                             
        self.init_hidden_decoder = torch.nn.Linear(in_features= self.latent_size, out_features= self.hidden_size * self.lstm_factor, device=device)
        self.decoder_lstm = torch.nn.LSTM(input_size= self.embed_size, hidden_size= self.hidden_size, batch_first = True, num_layers = self.num_layers)
        self.output = torch.nn.Linear(in_features= self.hidden_size * self.lstm_factor, out_features= self.vocab_size, device=device)
        self.softmax = torch.nn.Softmax(dim=2)

    def init_hidden(self, batch_size):
        hidden_cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        state_cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return (hidden_cell, state_cell)

    def get_embedding(self, x):
        x_embed = self.embed(x)
        maximum_sequence_length = x_embed.size(1)
        return x_embed, maximum_sequence_length

    def encoder(self, packed_x_embed,total_padding_length, hidden_encoder):
        # pad the packed input.
        packed_output_encoder, hidden_encoder = self.encoder_lstm(packed_x_embed, hidden_encoder)
        output_encoder, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output_encoder, batch_first=True, total_length= total_padding_length)

        # Extimate the mean and the variance of q(z|x)
        mean = self.mean(hidden_encoder[0])
        log_var = self.log_variance(hidden_encoder[0])
        std = torch.exp(0.5 * log_var)   # e^(0.5 log_var) = var^0.5
        
        # Generate a unit gaussian noise.
        batch_size = output_encoder.size(0)
        seq_len = output_encoder.size(1)
        noise = torch.randn(batch_size, self.latent_size).to(self.device)
        
        z = noise * std + mean

        return z, mean, log_var, hidden_encoder


    def decoder(self, z, packed_x_embed, total_padding_length=None):
        hidden_decoder = self.init_hidden_decoder(z)
        hidden_decoder = (hidden_decoder, hidden_decoder)

        # pad the packed input.
        packed_output_decoder, hidden_decoder = self.decoder_lstm(packed_x_embed,hidden_decoder) 
        output_decoder, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output_decoder, batch_first=True, total_length= total_padding_length)

        x_hat = self.output(output_decoder)

        # A trick to apply binary cross entropy by using cross entropy loss. 
        # neg_x_hat = (1 - x_hat)
            
        # binary_x_hat = torch.stack((x_hat, neg_x_hat), dim=3).contiguous()
        # print(binary_logits.size())
        # binary_x_hat = binary_x_hat.view(-1, 2)

        x_hat = self.softmax(x_hat)
        # x_hat = torch.flatten(x_hat)
        
        return (x_hat, hidden_decoder)

    def forward(self, x,sentences_length,hidden_encoder):
        """
        x : bsz * seq_len
        
        hidden_encoder: ( num_lstm_layers * bsz * hidden_size, num_lstm_layers * bsz * hidden_size)
        """
        # Get Embeddings
        x_embed, maximum_padding_length = self.get_embedding(x)
        # Packing the input
        packed_x_embed = torch.nn.utils.rnn.pack_padded_sequence(input= x_embed, lengths= sentences_length, batch_first=True, enforce_sorted=False)
        # Encoder
        z, mean, log_var, hidden_encoder = self.encoder(packed_x_embed, maximum_padding_length, hidden_encoder)
        # Decoder
        x_hat, _ = self.decoder(z, packed_x_embed, maximum_padding_length)
        
        return x_hat, mean, log_var, z, hidden_encoder


    def inference(self, n_samples, z, sos=None):
        # generate random z 
        sentences_length = torch.tensor([1])
        idx_sample = []

        if sos is None:
            x = torch.zeros(1,1,self.vocab_size).to(self.device)
            x[:,:,30] = 1

        hidden_decoder = self.init_hidden_decoder(z)
        hidden_decoder = (hidden_decoder, hidden_decoder)
        
        with torch.no_grad():
            for _ in range(n_samples):
                x_embed,max_sentence_length = self.get_embedding(x)
                # Packing the input
                packed_x_embed = torch.nn.utils.rnn.pack_padded_sequence(input= x_embed, lengths= sentences_length, batch_first=True, enforce_sorted=False)
                packed_output_decoder,hidden_decoder = self.decoder_lstm(packed_x_embed,hidden_decoder)
                output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output_decoder, batch_first=True, total_length= max_sentence_length)
                x_hat = self.output(output)

                x_hat = self.softmax(x_hat)

                max_id = torch.max(x_hat, 2)
                # sample = sample.squeeze().unsqueeze(0).unsqueeze(1) # (88,1) -> (1,1,88)
                l = [0 for i in range(88)]
                l[max_id[1].item()] = 1
                print(max_id[1].item())
                l = torch.tensor(l).unsqueeze(0).unsqueeze(1)
                
                idx_sample.append(l)

                x = l.float()

        note_samples = idx_sample
        note_samples = torch.stack(note_samples).squeeze(1).squeeze(1)
        return note_samples

class VAE_Loss(torch.nn.Module):
    def __init__(self, A: torch.Tensor):
        super(VAE_Loss, self).__init__()
        self.A = A
  
    def KL_loss (self, mu, log_var):
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl = kl.sum(-1)
        return kl.mean()
    
    def reconstruction_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        v = torch.abs(x - y)
        results = torch.bmm(torch.matmul(v, self.A), v.transpose(1, 2))
        results = F.normalize(results)
        results_sum = torch.sum(results, dim=(0, 1, 2))
        return results_sum * 10e-5

    def forward(self, mu, log_var, z, x_hat, x):
        kl_loss = self.KL_loss(mu, log_var)
        recon_loss = self.reconstruction_loss(x_hat, x)
        return kl_loss + recon_loss, kl_loss, recon_loss

class Trainer:
    def __init__(self, train_loader, test_loader, model, loss, optimizer, device="cpu") -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.print_interval = 10

    def train(self, train_losses, epoch, batch_size, clip) -> list:  
        states = self.model.init_hidden(batch_size)
        
        pbar = tqdm(enumerate(self.train_loader), total=21, desc=f"Epoch {epoch}")

        for batch_num, batch in pbar:
            # get the labels
            source, target, source_lengths = post_process_sequence_batch(batch)
            source = source.reshape(source.size(1), source.size(0), source.size(2)).to(self.device)
            target = target.to(self.device)
            source_lengths = torch.tensor(source_lengths)

            x_hat, mu, log_var, z, states = self.model(source,source_lengths, states)

            # Detach hidden states
            states = states[0].detach(), states[1].detach()

            # Compute the loss
            mloss, KL_loss, recon_loss = self.loss(mu = mu, log_var = log_var, z = z, x_hat = x_hat , x = target)

            # Update graphic
            train_losses.append((mloss , KL_loss.item(), recon_loss.item()))
            pbar.set_postfix_str(f"total : {mloss:.3g}, error : {recon_loss:.3g}, kl : {KL_loss:.3g}")
            
            # Backward the loss
            mloss.backward()

            # Clip the gradient
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            self.optimizer.step()

            self.optimizer.zero_grad()
            
        return train_losses

    def test(self, test_losses, epoch, batch_size) -> list:
        with torch.no_grad():
            states = self.model.init_hidden(batch_size) 

            for batch_num, batch in enumerate(self.test_loader): # loop over the data, and jump with step = bptt.
                # get the labels
                source, target, source_lengths = post_process_sequence_batch(batch)
                source = source.reshape(source.size(1), source.size(0), source.size(2)).to(self.device)
                target = target.to(self.device)
                source_lengths = torch.tensor(source_lengths)
                
                x_hat_param, mu, log_var, z, states = self.model(source,source_lengths, states)

                # detach hidden states
                states = states[0].detach(), states[1].detach()

                # compute the loss
                mloss, KL_loss, recon_loss = self.loss(mu = mu, log_var = log_var, z = z, x_hat_param = x_hat_param , x = target)

                test_losses.append((mloss , KL_loss.item(), recon_loss.item()))

                #Statistics.
                if batch_num % self.print_interval ==0:
                  print('| epoch {:3d} | elbo_loss {:5.6f} | kl_loss {:5.6f} | recons_loss {:5.6f} '.format(
                        epoch, mloss.item(), KL_loss.item(), recon_loss.item()))

            return test_losses

# Load the data
trainset = PianoGenerationDataset('./data/Nottingham/train/', longest_sequence_length=None)  #  Instead of None, use : model_setting["bptt"] to control the sequence length.
testset = PianoGenerationDataset('./data/Nottingham/test/', longest_sequence_length=None)

# Batchify the data
train_loader = torch.utils.data.DataLoader(trainset, batch_size=training_setting["batch_size"],shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(trainset, batch_size=training_setting["batch_size"],shuffle=True, drop_last=True)


vocab_size = model_setting['note_size']

model = LSTM_MUSIC_VAE(
    vocab_size = vocab_size,
    embed_size = model_setting["embed_size"],
    hidden_size = model_setting["hidden_size"],
    latent_size = model_setting["latent_size"],
    device = training_setting["device"]
).to(training_setting["device"])

Loss = VAE_Loss(torch.tensor([[1 if i == j else 0 for i in range(88)] for j in range(88)], dtype=torch.float))
optimizer = torch.optim.Adam(model.parameters(), lr=training_setting["lr"])
trainer = Trainer(train_loader, test_loader, model, Loss, optimizer, device=training_setting["device"])
model.load_state_dict(torch.load("models/VAE/first_test.pt"))

if __name__ == "__main__":
    # train_losses = []
    # test_losses = []
    # for epoch in tqdm(range(training_setting["epochs"])):
    #     train_losses = trainer.train(train_losses, epoch, training_setting["batch_size"], training_setting["clip"])
    #     # test_losses = trainer.test(test_losses, epoch, training_setting["batch_size"])

    # torch.save(model.state_dict(), f"models/VAE/first_test.pt")

    # plot_elbo(train_losses, "train")
    # plot_elbo(test_losses, "test")
    
    model.load_state_dict(torch.load("models/VAE/first_test.pt"))
    z = torch.randn(1, 1, model_setting["latent_size"]).to(training_setting["device"])
    sample = model.inference(500, z)
    print(sample)
    midiwrite("sample.midi", sample)