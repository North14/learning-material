#!/usr/bin/env python3

# The following code represents a project combining
# Autoencoder to a Spiking Neural Network (SNN) for semi-supervised learning.
# It is inspired by the work of Tavanaei et al. (2019) on deep learning in spiking neural networks.
#
# Autoencoder:
# Deep architecture (4 hidden)
# Using LeakyRelu activation function
# Input: Benign - random gaussian noise, Abnormal - random salt-and-pepper noise
# Output: Calcuate reconstruction loss between input and output per feature
# Store results in a reconstruction loss matrix
#
# Spiking Neural Network:
# Using Leaky Integrate-and-Fire (LIF) neuron model
# Training with surrogate gradient descent
# Input: Reconstruction loss matrix from Autoencoder
# Output: DDoS attack detection (binary classification)
# 

import torch
import torch.nn as nn
import torch.optim as optim

def gaussian_noise(data, mean=0.0, std=0.1):
    noise = torch.randn_like(data) * std + mean
    return data + noise

# Now testing on anomaly data (salt-and-pepper noise)
def salt_and_pepper_noise(data, prob=0.25):
    noisy_data = data.clone()
    rand = torch.rand_like(data)
    noisy_data[rand < prob / 2] = 0.0
    noisy_data[rand > 1 - prob / 2] = 1.0
    return noisy_data


class Autoencoder(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[64, 32, 8, 32]):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.LeakyReLU(True),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LeakyReLU(True),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.LeakyReLU(True),
            nn.Linear(hidden_sizes[2], hidden_sizes[3]),
            nn.LeakyReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_sizes[3], hidden_sizes[2]),
            nn.LeakyReLU(True),
            nn.Linear(hidden_sizes[2], hidden_sizes[1]),
            nn.LeakyReLU(True),
            nn.Linear(hidden_sizes[1], hidden_sizes[0]),
            nn.ReLU(True),
            nn.Linear(hidden_sizes[0], input_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

class SpikingNeuronLayer(nn.Module):
    def __init__(self, input_size, output_size, threshold=1.0, decay=0.9):
        super(SpikingNeuronLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.threshold = threshold
        self.decay = decay
        self.mem = None

    def forward(self, x):
        if self.mem is None:
            self.mem = torch.zeros(x.size(0), self.fc.out_features).to(x.device)
        # Update membrane potential
        self.mem = self.decay * self.mem + self.fc(x)
        # Generate spikes
        spikes = (self.mem >= self.threshold).float()
        # Reset membrane potential where spikes occurred
        self.mem = self.mem * (1 - spikes)
        return spikes

class Process:
    def __init__(self):
        self.r_matrix = None
        self.t_matrix = None
        self.threshold = 0.0
        self.s_matrix = None


    def reconstruction_matrix(self, x, x_recon):
        # Calculate absolute reconstruction loss per feature
        self.r_matrix = torch.abs(x - x_recon)
        return self.r_matrix
    
    def anomaly_threshold(self, k=1.0):
        # Threshold = Median + k * MAD
        if self.r_matrix is None:
            raise ValueError("Reconstruction matrix not computed yet.")
        median = torch.median(self.r_matrix, dim=0).values
        mad = torch.median(torch.abs(self.r_matrix - median), dim=0).values
        self.threshold = median + k * mad
        return self.threshold

    def signal_matrix(self):
        # Convert anomalies to binary signals based on threshold
        if self.r_matrix is None or self.threshold is None:
            raise ValueError("Reconstruction matrix or threshold not computed yet.")
        self.s_matrix = (self.r_matrix > self.threshold).float()
        return self.s_matrix
    

def main():
    input_size = 128
    autoencoder = Autoencoder(input_size=input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    # Dummy data
    data = torch.randn((64, input_size))  # Batch of 64 samples

    # Add noise
    noisy_data = gaussian_noise(data)

    # Train autoencoder
    for epoch in range(10):
        optimizer.zero_grad()
        reconstructed = autoencoder(noisy_data)
        loss = criterion(reconstructed, data)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

    # Get reconstruction loss matrix
    process_instance = Process()
    recon_matrix = process_instance.reconstruction_matrix(data, reconstructed)
    print("Reconstruction Loss Matrix:", recon_matrix)
    # Calculate anomaly threshold
    threshold = process_instance.anomaly_threshold(k=1.0)
    print("Anomaly Threshold:", threshold)
    # Generate signal matrix
    signal_matrix = process_instance.signal_matrix()
    print("Signal Matrix:", signal_matrix)

    # Spiking Neural Network for classification
    snn_layer = SpikingNeuronLayer(input_size=input_size, output_size=1)
    spikes = snn_layer(signal_matrix)
    # print("Spikes from SNN Layer (should be all 0):", spikes)

    anomaly_data = salt_and_pepper_noise(data)
    noisy_anomaly_data = gaussian_noise(anomaly_data)
    reconstructed_anomaly = autoencoder(noisy_anomaly_data)
    recon_matrix_anomaly = process_instance.reconstruction_matrix(anomaly_data, reconstructed_anomaly)
    signal_matrix_anomaly = process_instance.signal_matrix()
    spikes_anomaly = snn_layer(signal_matrix_anomaly)
    print("Spikes from SNN Layer on Anomaly Data (should have spikes):", spikes_anomaly)

    # create a seaborn plot visualization
    # y should be the reconstruction loss
    # x should be the sample index
    # Each sample should be represented as violin plot
    # color shoudld be red for anomaly data and blue for normal data
    # Create a vertical line for the threshold
    # create bars for the spike locations
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    normal_loss = recon_matrix.mean(dim=1).detach().numpy()
    anomaly_loss = recon_matrix_anomaly.mean(dim=1).detach().numpy()
    df_normal = pd.DataFrame({'Sample Index': range(len(normal_loss)), 'Reconstruction Loss': normal_loss, 'Type': 'Normal'})
    df_anomaly = pd.DataFrame({'Sample Index': range(len(anomaly_loss)), 'Reconstruction Loss': anomaly_loss, 'Type': 'Anomaly'})
    df = pd.concat([df_normal, df_anomaly])
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Sample Index', y='Reconstruction Loss', hue='Type', data=df, split=True, palette={'Normal': 'blue', 'Anomaly': 'red'})
    plt.axhline(y=threshold.mean().item(), color='green', linestyle='--', label='Anomaly Threshold')
    plt.title('Reconstruction Loss Distribution')
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()