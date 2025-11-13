# from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
# from lerobot.datasets.lerobot_dataset import LeRobotDataset

# # Streams directly from the Hugging Face Hub, without downloading the dataset into disk or loading into memory
# repo_id = "yaak-ai/L2D-v3"
# # repo_id = "physical-intelligence/libero"
# dataset = StreamingLeRobotDataset(repo_id)
# # dataset = LeRobotDataset(repo_id)

from datasets import load_dataset
from huggingface_hub import hf_hub_download

def copy_chunk(chunk_number:int):
    path = hf_hub_download(
        repo_id="commaai/comma2k19",
        repo_type="dataset",
        filename="raw_data/Chunk_1.zip"
    )

print(path)