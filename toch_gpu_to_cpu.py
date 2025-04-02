import torch

model = torch.load("model_best_weights.pt", map_location=torch.device('cpu'))

torch.save(model, "model_best_weights_cpu.pt")

print("Saved CPU-compatible model as model_best_weights_cpu.pt")