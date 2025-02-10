import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- Define PINN Model ---
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.hidden(x)

# --- Data Preparation ---
# Convert dBm to linear scale
def dBm_to_linear(dBm):
    return 10 ** (dBm / 10)

# Generate synthetic data
np.random.seed(42)
torch.manual_seed(42)

num_samples = 100
true_k = torch.randint(1, 5, (num_samples,)).float()
osnr_dB = torch.randn(num_samples) * 2 + 20
signal_power_dBm = torch.randn(num_samples) * 2 + -5
noise_power_dBm = torch.randn(num_samples) * 2 + -30

signal_power = torch.tensor(dBm_to_linear(signal_power_dBm))
noise_power = torch.tensor(dBm_to_linear(noise_power_dBm))

# Normalize inputs
osnr_normalized = (osnr_dB - torch.mean(osnr_dB)) / torch.std(osnr_dB)
signal_power_normalized = (signal_power - torch.mean(signal_power)) / torch.std(signal_power)
noise_power_normalized = (noise_power - torch.mean(noise_power)) / torch.std(noise_power)

inputs = torch.stack((osnr_normalized, signal_power_normalized, noise_power_normalized), dim=1)

# --- Define Physics-Informed Loss ---
def physics_loss(predicted_k, osnr_dB, signal_power, noise_power):
    predicted_osnr_linear = signal_power / (noise_power * predicted_k)
    predicted_osnr_dB = 10 * torch.log10(predicted_osnr_linear)
    mse_loss = nn.MSELoss()
    return mse_loss(predicted_osnr_dB, osnr_dB)

# --- Initialize Model, Optimizer, and Scheduler ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=500, verbose=True)

# Move tensors to device
inputs, true_k, osnr_dB, signal_power, noise_power = inputs.to(device), true_k.to(device), osnr_dB.to(device), signal_power.to(device), noise_power.to(device)

# --- Training Loop ---
num_epochs = 5000
loss_history = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    predicted_k = model(inputs)
    loss = physics_loss(predicted_k, osnr_dB, signal_power, noise_power)
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    loss_history.append(loss.item())
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

# --- Visualization ---
true_k_cpu = true_k.cpu().detach().numpy()
predicted_k_cpu = model(inputs).cpu().detach().numpy()

plt.scatter(true_k_cpu, predicted_k_cpu, color='blue', label="Predicted vs. True")
plt.plot([1, 4], [1, 4], 'r--', label="Ideal Fit")
plt.xlabel("True k")
plt.ylabel("Predicted k")
plt.title("Predicted vs. True k Values")
plt.legend()
plt.show()

# --- Model Evaluation ---
absolute_error = np.abs(true_k_cpu - predicted_k_cpu)
mean_absolute_error = np.mean(absolute_error)
print(f"Mean Absolute Error: {mean_absolute_error:.4f}")
