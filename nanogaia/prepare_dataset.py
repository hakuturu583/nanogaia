# from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
# from lerobot.datasets.lerobot_dataset import LeRobotDataset

# # Streams directly from the Hugging Face Hub, without downloading the dataset into disk or loading into memory
# repo_id = "yaak-ai/L2D-v3"
# # repo_id = "physical-intelligence/libero"
# dataset = StreamingLeRobotDataset(repo_id)
# # dataset = LeRobotDataset(repo_id)

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("commaai/comma2k19")
print(ds)
