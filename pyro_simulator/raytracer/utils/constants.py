import torch

UPWARDS = torch.tensor(1, dtype=torch.float32)
UPDOWN = torch.tensor(-1, dtype=torch.float32)
FARAWAY = torch.tensor(1.0e10, dtype=torch.float32)
SKYBOX_DISTANCE = torch.tensor(1.0e6, dtype=torch.float32)
