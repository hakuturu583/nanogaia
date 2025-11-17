"""
Cosmos-1.0-Tokenizer-CV8x8x8 + FlashAttention2 Transformer
Autoregressive latent video prediction model (240p).

- Input:   video_past  (B, 16, 3, 240, 320), normalized to [-1, 1]
- Input:   actions_past (B, 16, D_action_raw)
- Output:  video_future_pred (B, ~8, 3, 240, 320)

The VAE compresses time by 8×, so 16 frames become 2 latent steps.
Predicting 1 latent step corresponds to ~8 output frames.
"""

import torch
import torch.nn as nn
from diffusers import AutoencoderKLCosmos
from flash_attn.modules.mha import FlashMHA


# =========================================================
# 1. Cosmos Video Tokenizer Wrapper (CV8x8x8, 240p)
# =========================================================

class CosmosVideoTokenizer(nn.Module):
    """
    Wrapper around the Cosmos CV8x8x8 tokenizer.
    Handles conversion between (B, F, C, H, W) and
    (B, C_lat, T_lat, H_lat, W_lat).
    """

    def __init__(
        self,
        model_id: str = "nvidia/Cosmos-1.0-Tokenizer-CV8x8x8",
        subfolder: str = "vae",
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        super().__init__()
        self.vae = AutoencoderKLCosmos.from_pretrained(
            model_id,
            subfolder=subfolder,
            torch_dtype=dtype,
        ).to(device)
        self.vae.eval()
        self.dtype = dtype
        self.device = device

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode raw video frames into latent space.

        Args:
            video: (B, F, C, H, W), normalized to [-1, 1]

        Returns:
            latents: (B, C_lat, T_lat, H_lat, W_lat)
        """
        if video.dim() != 5:
            raise ValueError("video must be (B, F, C, H, W)")

        video = video.to(self.device, self.dtype)
        video_bcfhw = video.permute(0, 2, 1, 3, 4)  # (B, C, F, H, W)

        latents_dist = self.vae.encode(video_bcfhw)
        z = latents_dist.latent_dist.sample()
        return z

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent video representation back to frames.

        Args:
            latents: (B, C_lat, T_lat, H_lat, W_lat)

        Returns:
            video: (B, F, C, H, W), in [-1, 1]
        """
        latents = latents.to(self.device, self.dtype)
        recon = self.vae.decode(latents).sample  # (B, C, F, H, W)
        recon_bfchw = recon.permute(0, 2, 1, 3, 4)
        return recon_bfchw


# =========================================================
# 2. Action Embedding (aggregates frame-rate actions into latent-rate actions)
# =========================================================

class ActionAggregator(nn.Module):
    """
    Groups frame-level actions into latent-level actions.

    Example:
        16 frame actions → group by 8 → 2 latent actions
    """

    def __init__(self, action_dim_raw: int, group_size: int, out_dim: int, t_latent: int):
        super().__init__()
        self.group_size = group_size
        self.t_latent = t_latent

        self.mlp = nn.Sequential(
            nn.Linear(action_dim_raw * group_size, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: (B, F_in, D_raw)

        Returns:
            (B, T_latent, out_dim)
        """
        B, F, D = actions.shape
        G = self.t_latent
        needed = G * self.group_size

        # Pad or crop to match expected size
        if F < needed:
            pad_len = needed - F
            pad = actions[:, -1:, :].expand(B, pad_len, D)
            actions = torch.cat([actions, pad], dim=1)
        else:
            actions = actions[:, :needed, :]

        # Group frames
        actions = actions.view(B, G, self.group_size * D)
        out = self.mlp(actions)
        return out


class ActionEmbedding(nn.Module):
    """
    Combines aggregated actions with positional embeddings.
    """

    def __init__(self, action_dim_raw: int, group_size: int,
                 d_model: int, t_latent: int):
        super().__init__()
        self.agg = ActionAggregator(action_dim_raw, group_size, d_model, t_latent)
        self.pos = nn.Embedding(t_latent, d_model)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: (B, F_in, D_raw)

        Returns:
            (B, T_latent, d_model)
        """
        a = self.agg(actions)
        B, T, D = a.shape

        t_idx = torch.arange(T, device=actions.device)
        pos = self.pos(t_idx)[None, :, :]
        return a + pos


# =========================================================
# 3. LatentFlattener for CV8x8x8 (30×40 spatial, 1×1 patches)
# =========================================================

class LatentFlattener(nn.Module):
    """
    Converts latent (C, T, H, W) to tokens (N, C) and back.
    CV8x8x8 uses 1×1 patches → tokens correspond to cells.
    """

    def __init__(self, c_latent: int = 16, height: int = 30, width: int = 40):
        super().__init__()
        self.c_latent = c_latent
        self.h = height
        self.w = width
        self.c_tok = c_latent

    def latent_to_tokens(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, C_lat, T, H, W)

        Returns:
            tokens: (B, T*H*W, C_tok)
        """
        B, C, T, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
        return z_flat

    def tokens_to_latent(self, tokens: torch.Tensor, t_out: int) -> torch.Tensor:
        """
        Args:
            tokens: (B, t_out * H * W, C_tok)

        Returns:
            (B, C_lat, t_out, H, W)
        """
        B, N, C = tokens.shape
        H, W = self.h, self.w
        z = tokens.view(B, t_out, H, W, C)
        z = z.permute(0, 4, 1, 2, 3)
        return z


# =========================================================
# 4. TokenFuser + FlashAttention2 Decoder
# =========================================================

class TokenFuser(nn.Module):
    """
    Fuses video tokens and action embeddings into decoder input tokens.
    """

    def __init__(self, c_tok: int, d_model: int,
                 t_latent: int, h_tok: int, w_tok: int):
        super().__init__()
        self.video_proj = nn.Linear(c_tok, d_model // 2)
        self.action_proj = nn.Linear(d_model, d_model // 2)
        self.out_proj = nn.Linear(d_model, d_model)

        self.t_latent = t_latent
        self.h_tok = h_tok
        self.w_tok = w_tok

    def forward(self, z_tokens: torch.Tensor,
                a_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_tokens: (B, N, C_tok)
            a_emb:    (B, T, d_model)

        Returns:
            (B, N, d_model)
        """
        B, N, _ = z_tokens.shape
        T = self.t_latent
        Ht, Wt = self.h_tok, self.w_tok
        assert N == T * Ht * Wt

        z = self.video_proj(z_tokens)

        # Expand action embeddings across spatial cells
        a = a_emb.unsqueeze(2)
        a = a.expand(B, T, Ht * Wt, a.size(-1))
        a = a.reshape(B, N, -1)
        a = self.action_proj(a)

        tok = torch.cat([z, a], dim=-1)
        tok = self.out_proj(tok)
        return tok


class FlashDecoderLayer(nn.Module):
    """
    Single layer of a causal Transformer decoder using FlashAttention2.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = FlashMHA(
            embed_dim=d_model,
            num_heads=n_heads,
            attention_dropout=dropout,
            causal=True,  # prevents attending to future tokens
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.self_attn(h)
        x = x + h

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h

        return x


class FlashDecoder(nn.Module):
    """
    A Transformer decoder made of multiple FlashDecoderLayers.
    """

    def __init__(self, d_model: int, n_heads: int, num_layers: int,
                 d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            FlashDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# =========================================================
# 5. AR Core (latent → latent) + Full Model
# =========================================================

class VideoARTCoreCV8x8x8(nn.Module):
    """
    Autoregressive latent predictor.
    Converts past latent frames + past actions
    → predicts the next latent frame.
    """

    def __init__(
        self,
        c_latent: int = 16,
        h_latent: int = 30,
        w_latent: int = 40,
        t_in_latent: int = 2,
        frames_per_latent: int = 8,
        action_dim_raw: int = 10,
        d_model: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
    ):
        super().__init__()
        self.t_in_latent = t_in_latent
        self.frames_per_latent = frames_per_latent

        self.flattener = LatentFlattener(
            c_latent=c_latent,
            height=h_latent,
            width=w_latent,
        )

        self.action_emb = ActionEmbedding(
            action_dim_raw=action_dim_raw,
            group_size=frames_per_latent,
            d_model=d_model,
            t_latent=t_in_latent,
        )

        self.fuser = TokenFuser(
            c_tok=self.flattener.c_tok,
            d_model=d_model,
            t_latent=t_in_latent,
            h_tok=h_latent,
            w_tok=w_latent,
        )

        self.decoder = FlashDecoder(
            d_model=d_model,
            n_heads=num_heads,
            num_layers=num_layers,
            d_ff=dim_feedforward,
        )

        self.to_latent_tok = nn.Linear(d_model, self.flattener.c_tok)

    def forward(self, z_past: torch.Tensor,
                actions_past: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_past:       (B, C_lat, T_in, H, W)
            actions_past: (B, F_in, D_action_raw)

        Returns:
            z_future_pred: (B, C_lat, 1, H, W)
        """
        B, C, T, H, W = z_past.shape
        assert T == self.t_in_latent

        z_tokens = self.flattener.latent_to_tokens(z_past)    # (B, N, C_tok)

        a_emb = self.action_emb(actions_past)                 # (B, T, d_model)

        tok = self.fuser(z_tokens, a_emb)                     # (B, N, d_model)

        h = self.decoder(tok)                                 # (B, N, d_model)

        # Extract tokens corresponding to final timestep
        N_per_t = H * W
        last_block = h[:, -N_per_t:, :]                       # (B, H*W, d_model)

        last_latent_tok = self.to_latent_tok(last_block)      # (B, H*W, C_tok)

        z_future = self.flattener.tokens_to_latent(last_latent_tok, t_out=1)
        return z_future


class CosmosVideoARModel(nn.Module):
    """
    Full model wrapping:
    - Cosmos Video Tokenizer
    - AR latent predictor core
    """

    def __init__(
        self,
        tokenizer: CosmosVideoTokenizer,
        c_latent: int = 16,
        h_latent: int = 30,
        w_latent: int = 40,
        t_in_latent: int = 2,
        frames_per_latent: int = 8,
        action_dim_raw: int = 10,
        d_model: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.core = VideoARTCoreCV8x8x8(
            c_latent=c_latent,
            h_latent=h_latent,
            w_latent=w_latent,
            t_in_latent=t_in_latent,
            frames_per_latent=frames_per_latent,
            action_dim_raw=action_dim_raw,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
        )

    def forward(self, video_past: torch.Tensor,
                actions_past: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_past:   (B, F_in, C, H, W)
            actions_past: (B, F_in, D_action_raw)

        Returns:
            video_future_pred: (B, F_out, C, H, W)
        """
        z_past = self.tokenizer.encode(video_past)

        z_future = self.core(z_past, actions_past)

        video_future = self.tokenizer.decode(z_future)
        return video_future


# =========================================================
# 6. Demo
# =========================================================

if __name__ == "__main__":
    B = 1
    F_in = 16
    H, W = 240, 320
    D_action_raw = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = CosmosVideoTokenizer(
        model_id="nvidia/Cosmos-1.0-Tokenizer-CV8x8x8",
        subfolder="vae",
        dtype=torch.float16,
        device=device,
    )

    model = CosmosVideoARModel(
        tokenizer=tokenizer,
        c_latent=16,
        h_latent=30,
        w_latent=40,
        t_in_latent=2,          
        frames_per_latent=8,    
        action_dim_raw=D_action_raw,
        d_model=512,
        num_layers=8,
        num_heads=8,
        dim_feedforward=2048,
    ).to(device)

    model.eval()

    # Dummy inputs
    video_past = torch.randn(B, F_in, 3, H, W)
    video_past = torch.clamp(video_past, -1, 1)

    actions_past = torch.randn(B, F_in, D_action_raw).to(device)

    with torch.no_grad():
        video_future_pred = model(video_past, actions_past)

    print("video_future_pred shape:", video_future_pred.shape)
