import argparse
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from nanogaia.latent_dataset import LatentDataset
from nanogaia.model import CosmosVideoTokenizer, VideoARTCoreCV8x8x8

import matplotlib.pyplot as plt

LOG_2_PI_E = math.log(2 * math.pi * math.e)


@dataclass
class WandbConfig:
    project: str = "nanogaia"
    run_name: str | None = None
    disable: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WandbConfig":
        base = cls()
        return cls(
            project=data.get("project", base.project),
            run_name=data.get("run_name", base.run_name),
            disable=bool(data.get("disable", base.disable)),
        )


@dataclass
class NetworkConfig:
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 2
    dtype: str | None = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetworkConfig":
        base = cls()
        raw_dtype = data.get("dtype", base.dtype)
        dtype = str(raw_dtype).lower() if raw_dtype is not None else None
        return cls(
            d_model=int(data.get("d_model", base.d_model)),
            num_layers=int(data.get("num_layers", base.num_layers)),
            num_heads=int(data.get("num_heads", base.num_heads)),
            dtype=dtype,
        )


@dataclass
class TrainConfig:
    lmdb_path: Path = Path(__file__).resolve().parent / "dataset" / "latent.lmdb"
    overfit_test: bool = False
    batch_size: int = 1
    num_workers: int = 0
    epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    log_interval: int = 10
    video_interval: int = 200
    video_fps: int = 4
    max_steps: int = 0
    device: str | None = None
    loss_mse_weight: float = 0.8
    loss_mae_weight: float = 0.2
    loss_scale: float = 1.0
    wandb: WandbConfig = field(default_factory=WandbConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], base_path: Path | None = None
    ) -> "TrainConfig":
        base = cls()
        wandb_cfg = WandbConfig.from_dict(data.get("wandb", {}))
        network_cfg = NetworkConfig.from_dict(data.get("network", {}))

        lmdb_path_raw = data.get("lmdb_path", base.lmdb_path)
        lmdb_path = Path(lmdb_path_raw)
        if base_path and not lmdb_path.is_absolute():
            lmdb_path = (base_path / lmdb_path).resolve()

        return cls(
            lmdb_path=lmdb_path,
            overfit_test=bool(data.get("overfit_test", base.overfit_test)),
            batch_size=int(data.get("batch_size", base.batch_size)),
            num_workers=int(data.get("num_workers", base.num_workers)),
            epochs=int(data.get("epochs", base.epochs)),
            lr=float(data.get("lr", base.lr)),
            weight_decay=float(data.get("weight_decay", base.weight_decay)),
            log_interval=int(data.get("log_interval", base.log_interval)),
            video_interval=int(data.get("video_interval", base.video_interval)),
            video_fps=int(data.get("video_fps", base.video_fps)),
            max_steps=int(data.get("max_steps", base.max_steps)),
            device=data.get("device", base.device),
            loss_mse_weight=float(data.get("loss_mse_weight", base.loss_mse_weight)),
            loss_mae_weight=float(data.get("loss_mae_weight", base.loss_mae_weight)),
            loss_scale=float(data.get("loss_scale", base.loss_scale)),
            wandb=wandb_cfg,
            network=network_cfg,
        )

    def to_log_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["lmdb_path"] = str(self.lmdb_path)
        return payload


def load_train_config(path: Path) -> TrainConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as handle:
        raw = yaml.safe_load(handle) or {}
    return TrainConfig.from_dict(raw, base_path=path.parent)


def apply_cli_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    if args.device:
        config.device = args.device
    if args.wandb_run_name:
        config.wandb.run_name = args.wandb_run_name
    if args.disable_wandb:
        config.wandb.disable = True
    return config


def to_tensor(sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v) for k, v in sample.items()}


def make_dataloader(
    lmdb_path: Path, batch_size: int, num_workers: int, overfit_test: bool
) -> DataLoader:
    dataset = LatentDataset(lmdb_path, transform=to_tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not overfit_test,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def resolve_dtype(device: torch.device, requested: str | None) -> torch.dtype:
    """
    Return torch dtype for the given device honoring a requested string.
    """
    if requested is None or requested == "auto":
        if device.type == "cuda":
            use_bf16 = torch.cuda.is_bf16_supported()
            return torch.bfloat16 if use_bf16 else torch.float16
        return torch.float32

    normalized = requested.lower()
    if normalized in ("float32", "fp32", "f32"):
        return torch.float32
    if normalized in ("float16", "fp16", "f16", "half"):
        if device.type == "cpu":
            print("float16 requested on CPU; falling back to float32.")
            return torch.float32
        return torch.float16
    if normalized in ("bfloat16", "bf16"):
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            print("bfloat16 requested but not supported on this GPU; falling back to float16.")
            return torch.float16
        return torch.bfloat16

    raise ValueError(
        f"Unknown dtype {requested!r}; expected one of float32, float16, bfloat16, or auto."
    )


def prepare_batch(
    batch: dict, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, ...]:
    """
    Returns:
        z_past:         (B, T_lat, C_lat, H_lat, W_lat)
        z_future:       (B, T_lat, C_lat, H_lat, W_lat)
        actions_past:   (B, 8, 3)
        actions_future: (B, 8, 3)
    """
    latents_past = batch["latent_past"].to(device=device, dtype=dtype)
    latents_future = batch["latent_future"].to(device=device, dtype=dtype)
    actions_past = batch["actions_past"].to(device=device, dtype=dtype)
    actions_future = batch["actions_future"].to(device=device, dtype=dtype)

    return latents_past, latents_future, actions_past, actions_future


def build_model(
    device: torch.device, dtype: torch.dtype, network: NetworkConfig
) -> VideoARTCoreCV8x8x8:
    model = VideoARTCoreCV8x8x8(
        t_in_latent=8,
        c_latent=2,
        frames_per_latent=8,
        action_dim_raw=3,
        d_model=network.d_model,
        num_layers=network.num_layers,
        num_heads=network.num_heads,
        dim_feedforward=512,
        t_future_latent=16,
        gradient_checkpointing=True,
    ).to(device=device, dtype=dtype)
    return model


def iterate_batches(
    dataloader: DataLoader, overfit_test: bool
) -> Iterator[dict]:
    if not overfit_test:
        yield from dataloader
        return

    try:
        first_batch = next(iter(dataloader))
    except StopIteration as exc:
        raise ValueError(
            "Dataloader is empty; cannot run overfit_test mode."
        ) from exc

    steps_per_epoch = len(dataloader)
    if steps_per_epoch == 0:
        raise ValueError("Dataloader is empty; cannot run overfit_test mode.")

    for _ in range(steps_per_epoch):
        yield first_batch


def average_grad_norm(model: torch.nn.Module) -> float:
    total_norm = 0.0
    count = 0
    for param in model.parameters():
        if param.grad is None:
            continue
        total_norm += param.grad.detach().float().norm().item()
        count += 1
    if count == 0:
        return 0.0
    return total_norm / count


def wandb_log_delta_latent(
    wandb_module: Any,
    z_past: torch.Tensor,
    z_future: torch.Tensor,
    step: int | None = None,
    tag: str = "delta_latent",
) -> Dict[str, float]:
    """
    Compute delta = z_future - z_past and log stats + histogram to wandb.
    """

    t = min(z_past.shape[2], z_future.shape[2])
    z_past = z_past[:, :, :t]
    z_future = z_future[:, :, :t]

    delta = (z_future - z_past).reshape(-1).detach().float().cpu()

    stats = {
        f"{tag}/mean": delta.mean().item(),
        f"{tag}/std": delta.std().item(),
        f"{tag}/max": delta.max().item(),
        f"{tag}/min": delta.min().item(),
        f"{tag}/abs_max": delta.abs().max().item(),
    }

    wandb_module.log(stats, step=step)

    hist = wandb_module.Histogram(delta.numpy(), num_bins=100)
    wandb_module.log({f"{tag}/hist": hist}, step=step)

    return stats


def train(args: argparse.Namespace) -> None:
    config = load_train_config(args.config)
    config = apply_cli_overrides(config, args)

    device = torch.device(
        config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dtype = resolve_dtype(device, config.network.dtype)

    use_wandb = not config.wandb.disable
    wandb_run = None
    wandb = None
    if use_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is required for logging; install with `pip install wandb` or pass --disable-wandb."
            ) from exc
        default_run_name = config.wandb.run_name
        if default_run_name is None:
            from datetime import datetime, timezone

            default_run_name = datetime.now(timezone.utc).isoformat()
        wandb_run = wandb.init(
            project=config.wandb.project,
            name=default_run_name,
            config=config.to_log_dict(),
        )

    dataloader = make_dataloader(
        config.lmdb_path, config.batch_size, config.num_workers, config.overfit_test
    )
    model = build_model(device, dtype, config.network)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scaler_enabled = device.type == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    amp_enabled = device.type == "cuda"

    tokenizer_for_logging = None
    if use_wandb and config.video_interval > 0:
        tokenizer_for_logging = CosmosVideoTokenizer(
            device=device,
            dtype=dtype,
        )

    global_step = 0
    for epoch in range(config.epochs):
        for batch in iterate_batches(dataloader, config.overfit_test):
            z_past, z_future, actions_past, actions_future = prepare_batch(
                batch, device, dtype
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=dtype, enabled=amp_enabled
            ):
                delta_future = model(z_past, actions_past, actions_future)
                pred_z = delta_future + z_past
                target_z = z_future

                loss_mse = F.mse_loss(pred_z, target_z)
                loss_mae = F.l1_loss(pred_z, target_z)
                mixed_loss = (
                    config.loss_mse_weight * loss_mse
                    + config.loss_mae_weight * loss_mae
                )
                loss = config.loss_scale * mixed_loss

                var = pred_z.float().var(
                    dim=tuple(range(1, pred_z.dim())), unbiased=False
                )
                entropy = 0.5 * (LOG_2_PI_E + torch.log(var.clamp_min(1e-12)))
                entropy_scalar = entropy.mean().detach()

            if scaler_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = average_grad_norm(model)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = average_grad_norm(model)
                optimizer.step()

            if global_step % config.log_interval == 0:
                print(
                    f"epoch {epoch} step {global_step} "
                    f"loss {loss.item():.4f} mse {loss_mse.item():.4f} "
                    f"mae {loss_mae.item():.4f} entropy {entropy_scalar.item():.4f} "
                    f"grad_norm {grad_norm:.4f} "
                    f"device {device} dtype {dtype}"
                )
                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/loss_mse": loss_mse.item(),
                            "train/loss_mae": loss_mae.item(),
                            "train/entropy": entropy_scalar.item(),
                            "train/loss_scale": config.loss_scale,
                            "train/epoch": epoch,
                            "train/avg_grad_norm": grad_norm,
                        },
                        step=global_step,
                    )
                    wandb_log_delta_latent(
                        wandb,
                        z_past.detach(),
                        z_future.detach(),
                        step=global_step,
                    )

            if (
                use_wandb
                and config.video_interval > 0
                and global_step > 0
                and global_step % config.video_interval == 0
                and tokenizer_for_logging is not None
            ):
                with torch.no_grad():
                    pred_fd, pred_path = tempfile.mkstemp(suffix=".gif")
                    tgt_fd, tgt_path = tempfile.mkstemp(suffix=".gif")
                    source_fd, source_video = tempfile.mkstemp(suffix=".gif")
                    os.close(pred_fd)
                    os.close(tgt_fd)
                    os.close(source_fd)
                    tokenizer_for_logging.decode_as_video(
                        delta_future[0] + z_past[0], pred_path, fps=config.video_fps
                    )
                    tokenizer_for_logging.decode_as_video(
                        z_future[0], tgt_path, fps=config.video_fps
                    )
                    tokenizer_for_logging.decode_as_video(
                        z_past[0], source_video, fps=config.video_fps
                    )
                wandb.log(
                    {
                        "train/input_video": wandb.Video(
                            source_video,
                            caption="input frames",
                            format="gif",
                        ),
                        "train/predicted_future": wandb.Video(
                            pred_path,
                            caption="predicted future frames",
                            format="gif",
                        ),
                        "train/target_future": wandb.Video(
                            tgt_path,
                            caption="target future frames",
                            format="gif",
                        ),
                    },
                    step=global_step,
                )
                os.remove(pred_path)
                os.remove(tgt_path)
                os.remove(source_video)

            global_step += 1
            if config.max_steps and global_step >= config.max_steps:
                print("Reached max_steps, stopping training loop.")
                if wandb_run:
                    wandb_run.finish()
                return

    if wandb_run:
        wandb_run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train VideoARTCoreCV8x8x8 on latent LMDB windows."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config defining training hyperparameters (e.g., nanogaia/train.yaml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from config (e.g., cuda:0 or cpu).",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Override W&B run name specified in config.",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Force disable wandb logging regardless of config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        train(parse_args())
    except KeyboardInterrupt:
        print("Interrupted by user, exiting.", file=sys.stderr)
