import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "yaak-ai/L2D-v3"

# 1) Load from the Hub (cached locally)
dataset = LeRobotDataset(repo_id)

# 2) Random access by index
sample = dataset[100]
print(sample)
# {
#   'observation.state': tensor([...]),
#   'action': tensor([...]),
#   'observation.images.front_left': tensor([C, H, W]),
#   'timestamp': tensor(1.234),
#   ...
# }

# 3) Temporal windows via delta_timestamps (seconds relative to t)
delta_timestamps = {
    "observation.images.front_left": [-0.2, -0.1, 0.0]  # 0.2s and 0.1s before current frame
}

dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

# Accessing an index now returns a stack for the specified key(s)
sample = dataset[100]
print(sample["observation.images.front_left"].shape)  # [T, C, H, W], where T=3

# 4) Wrap with a DataLoader for training
batch_size = 16
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
for batch in data_loader:
    observations = batch["observation.state"].to(device)
    actions = batch["action"].to(device)
    images = batch["observation.images.front_left"].to(device)
    # model.forward(batch)
