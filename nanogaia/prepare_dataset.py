# from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
# from lerobot.datasets.lerobot_dataset import LeRobotDataset

# # Streams directly from the Hugging Face Hub, without downloading the dataset into disk or loading into memory
# repo_id = "yaak-ai/L2D-v3"
# # repo_id = "physical-intelligence/libero"
# dataset = StreamingLeRobotDataset(repo_id)
# # dataset = LeRobotDataset(repo_id)

from datasets import load_dataset
from huggingface_hub import hf_hub_download
import os
import zipfile


def unzip_skip_existing(zip_path, dst_dir):
    """
    Unzip a ZIP archive, skipping files that already exist in the destination directory.
    Existing files are not overwritten.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            # Full path where the file will be extracted
            extracted_path = os.path.join(dst_dir, member.filename)

            # If this is a directory entry, ensure it exists and continue
            if member.is_dir():
                os.makedirs(extracted_path, exist_ok=True)
                continue

            # Skip extraction if the file already exists
            if os.path.exists(extracted_path):
                print(f"Skip (exists): {extracted_path}")
                continue

            # Make sure the parent directory exists
            os.makedirs(os.path.dirname(extracted_path), exist_ok=True)

            # Extract the file manually
            print(f"Extract: {extracted_path}")
            with z.open(member) as src, open(extracted_path, "wb") as dst:
                dst.write(src.read())


def prepare_dataset(chunk_number: int):
    def copy_chunk(chunk_number: int):
        path = hf_hub_download(
            repo_id="commaai/comma2k19",
            repo_type="dataset",
            filename="raw_data/Chunk_" + str(chunk_number) + ".zip",
        )
        return path

    copy_from = copy_chunk(chunk_number)
    copy_to = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    unzip_skip_existing(copy_from, copy_to)


for i in range(1, 11):
    prepare_dataset(i)
