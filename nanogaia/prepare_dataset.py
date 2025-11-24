# from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
# from lerobot.datasets.lerobot_dataset import LeRobotDataset

# # Streams directly from the Hugging Face Hub, without downloading the dataset into disk or loading into memory
# repo_id = "yaak-ai/L2D-v3"
# # repo_id = "physical-intelligence/libero"
# dataset = StreamingLeRobotDataset(repo_id)
# # dataset = LeRobotDataset(repo_id)

from huggingface_hub import hf_hub_download
import argparse
import cv2
import os
import zipfile
from pathlib import Path

from nanogaia.comma2k19_dataset import CommaDataset


def convert_hevc_to_jpeg_sequence(
    input_path: str, output_dir: str, target_height: int = 240, jpeg_quality: int = 90
):
    """
    Convert an HEVC video into a sequence of resized JPEG images.

    Args:
        input_path (str): Path to the input HEVC video.
        output_dir (str): Directory where output JPEG frames will be saved.
        target_height (int): Height of the resized frames (keeps aspect ratio).
        jpeg_quality (int): JPEG quality (0–100).
    """

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Get original size
        h, w = frame.shape[:2]

        def round_to_multiple_of_8(x):
            return round(x / 8) * 8

        target_height = round_to_multiple_of_8(target_height)

        scale = target_height / h
        target_width = int(w * scale)

        target_width = round_to_multiple_of_8(target_width)

        target_width = max(8, target_width)
        target_height = max(8, target_height)

        # Resize frame
        resized = cv2.resize(
            frame, (target_width, target_height), interpolation=cv2.INTER_AREA
        )

        # Save as JPEG
        out_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(out_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

        frame_idx += 1

    cap.release()
    print(f"Total frames exported: {frame_idx}")


def unzip_skip_existing(copy_from: str, copy_to: str):
    """
    Unzip / handle input while skipping files that already exist.

    Special case:
        If `copy_from` has extension `.hevc`,
        convert the HEVC video into a sequence of JPEG images
        under: <copy_to>/raw_images/

    Args:
        copy_from (str): Path to the source file (zip, hevc, etc.).
        copy_to (str): Destination directory.
    """
    src = Path(copy_from)
    dst = Path(copy_to)
    dst.mkdir(parents=True, exist_ok=True)
    print(f"Processing: {src} -> {dst}")

    # Example: if it's a zip file, unzip while skipping existing files
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src, "r") as zf:
            for member in zf.infolist():
                # Build destination path
                out_path = dst / member.filename

                print(f"Extracting: {out_path}")

                # Special handling for HEVC -> JPEG sequence
                if out_path.suffix.lower() == ".hevc":
                    convert_hevc_to_jpeg_sequence(
                        str(out_path), str(out_path.parent / "raw_images")
                    )

                # Skip if it already exists
                if out_path.exists():
                    continue

                # Create parent directories if needed
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # Extract this member
                with (
                    zf.open(member, "r") as source_fp,
                    open(out_path, "wb") as target_fp,
                ):
                    shutil.copyfileobj(source_fp, target_fp)
        return


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


def export_latent_lmdb(
    dataset_dir: Path, lmdb_name: str = "latent.lmdb", downsample_fps: int | None = 5
) -> None:
    """
    Build LMDB containing 8-frame past/future video and actions from the prepared dataset.
    """
    if downsample_fps is not None and downsample_fps <= 0:
        raise ValueError("downsample_fps must be positive or None.")

    downsample_stride = 1
    window_size = CommaDataset.TARGET_LATENT_FRAMES
    if downsample_fps:
        downsample_stride = max(
            1, int(round(CommaDataset.SOURCE_FPS / downsample_fps))
        )
        window_size = CommaDataset.TARGET_LATENT_FRAMES * downsample_stride

    dataset = CommaDataset(dataset_dir, window_size=window_size)
    lmdb_path = dataset_dir / lmdb_name
    dataset.export_as_latent_data(lmdb_path, downsample_stride=downsample_stride)
    effective_fps = CommaDataset.SOURCE_FPS / downsample_stride
    print(
        f"Wrote latent LMDB to: {lmdb_path} "
        f"(downsampled to ~{effective_fps:.2f} FPS)"
    )


def main():
    # for i in range(1, 11):
    #     prepare_dataset(i)
    dataset_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "dataset"
    export_latent_lmdb(dataset_dir)


if __name__ == "__main__":
    main()
