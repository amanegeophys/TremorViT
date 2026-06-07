import math
from pathlib import Path
from typing import TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..config.experiment_config import ExperimentConfig

AttentionMaps: TypeAlias = list[torch.Tensor | None]


class ConvPatchEmbedding(nn.Module):
    """Project waveform patches into token embeddings."""

    def __init__(self, emb_dim: int, patch_size: int, stride: int, num_input: int) -> None:
        """Initialize the convolutional patch embedding layer.

        Parameters
        ----------
        emb_dim : int
            Embedding dimension.
        patch_size : int
            One-dimensional convolution kernel size.
        stride : int
            One-dimensional convolution stride.
        num_input : int
            Number of input waveform channels.
        """
        super(ConvPatchEmbedding, self).__init__()
        self.conv = nn.Conv1d(num_input, emb_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed input waveform patches.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(batch, channels, samples)``.

        Returns
        -------
        torch.Tensor
            Patch embeddings with shape ``(batch, emb_dim, patches)``.
        """
        x = self.conv(x)
        return x


class PositionalEncoding(nn.Module):
    """Add positional embeddings to token sequences."""

    def __init__(self, seq_len: int, emb_dim: int, position_emb_type: str) -> None:
        """Initialize positional embeddings.

        Parameters
        ----------
        seq_len : int
            Sequence length.
        emb_dim : int
            Embedding dimension.
        position_emb_type : str
            Positional embedding type, either ``"learnable"`` or ``"sinusoidal"``.
        """
        super(PositionalEncoding, self).__init__()
        if position_emb_type == "learnable":
            self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, emb_dim))
        elif position_emb_type == "sinusoidal":
            pos_emb = torch.zeros(1, seq_len, emb_dim)
            position = torch.arange(0, seq_len).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, emb_dim, 2) * -(math.log(10000.0) / emb_dim)
            )
            pos_emb[0, :, 0::2] = torch.sin(position * div_term)
            pos_emb[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pos_embedding", pos_emb)
        else:
            raise ValueError(f"Unknown {position_emb_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to input tokens.

        Parameters
        ----------
        x : torch.Tensor
            Input token tensor.

        Returns
        -------
        torch.Tensor
            Position-encoded token tensor.
        """
        return x + self.pos_embedding


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention block."""

    def __init__(
        self, emb_dim: int, num_head: int, dropout_rate: float, save_attention: bool = False
    ) -> None:
        """Initialize multi-head attention.

        Parameters
        ----------
        emb_dim : int
            Embedding dimension.
        num_head : int
            Number of attention heads.
        dropout_rate : float
            Dropout probability.
        save_attention : bool, default=False
            Whether to keep the last attention map.
        """
        super().__init__()
        self.save_attention = save_attention
        self.num_head = num_head
        self.emb_dim = emb_dim
        if emb_dim % num_head != 0:
            raise ValueError(
                f"emb_dim ({emb_dim}) must be divisible by num_head ({num_head})"
            )
        self.head_dim = emb_dim // num_head

        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_o = nn.Linear(emb_dim, emb_dim)
        self.last_attention = None

        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention to a token sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tokens with shape ``(batch, seq_len, emb_dim)``.

        Returns
        -------
        torch.Tensor
            Attention output with the same shape as ``x``.
        """
        B, S, _ = x.size()
        q = self.w_q(x).view(B, S, self.num_head, self.head_dim)
        k = self.w_k(x).view(B, S, self.num_head, self.head_dim)
        v = self.w_v(x).view(B, S, self.num_head, self.head_dim)

        q = q.permute(0, 2, 1, 3)  # [B,H,S,Dh]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scale = self.head_dim**-0.5
        dots = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(dots, dim=-1)
        if self.save_attention:
            self.last_attention = attn.detach()
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, S, self.emb_dim)
        out = self.proj_drop(self.w_o(out))
        return out


class MLP(nn.Module):
    """Feed-forward network used inside the transformer encoder."""

    def __init__(self, emb_dim: int, feedforward_dim: int, dropout_rate: float) -> None:
        """Initialize the feed-forward network.

        Parameters
        ----------
        emb_dim : int
            Embedding dimension.
        feedforward_dim : int
            Hidden dimension.
        dropout_rate : float
            Dropout probability.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feedforward_dim, emb_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Transformed tensor.
        """
        return self.net(x)


class MixFFN(nn.Module):
    """Convolutional feed-forward network for token mixing."""

    def __init__(self, emb_dim: int, feedforward_dim: int, dropout_rate: float) -> None:
        """Initialize the convolutional feed-forward network.

        Parameters
        ----------
        emb_dim : int
            Embedding dimension.
        feedforward_dim : int
            Hidden dimension.
        dropout_rate : float
            Dropout probability.
        """
        super().__init__()
        self.linear1 = nn.Conv1d(
            in_channels=emb_dim,
            out_channels=feedforward_dim,
            kernel_size=1,
            stride=1,
        )
        self.conv = nn.Conv1d(
            in_channels=feedforward_dim,
            out_channels=feedforward_dim,
            kernel_size=3,
            groups=feedforward_dim,  # depthwise
            padding="same",
        )
        self.activation = nn.GELU()
        self.linear2 = nn.Conv1d(
            in_channels=feedforward_dim,
            out_channels=emb_dim,
            kernel_size=1,
            stride=1,
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional feed-forward network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(batch, seq_len, emb_dim)``.

        Returns
        -------
        torch.Tensor
            Output tensor with shape ``(batch, seq_len, emb_dim)``.
        """
        x = rearrange(x, "B L C -> B C L")
        x = self.linear1(x)
        x = self.conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = rearrange(x, "B C L -> B L C")
        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder blocks."""

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        feedforward_dim: int,
        num_layers: int,
        mlp_type: str,
        save_attention: bool = False,
    ) -> None:
        """Initialize transformer encoder layers.

        Parameters
        ----------
        emb_dim : int
            Embedding dimension.
        num_heads : int
            Number of attention heads.
        dropout : float
            Dropout probability.
        feedforward_dim : int
            Feed-forward hidden dimension.
        num_layers : int
            Number of encoder layers.
        mlp_type : str
            Feed-forward block type.
        save_attention : bool, default=False
            Whether attention maps should be stored.
        """
        super().__init__()
        MLP_MAP = {
            "MLP": MLP,
            "MixFFN": MixFFN,
        }
        try:
            mlp_cls = MLP_MAP[mlp_type]
        except KeyError:
            raise ValueError(
                f"Unknown mlp_type={mlp_type}. Choose from {list(MLP_MAP)}"
            )

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "layer_norm1": nn.LayerNorm(emb_dim),
                        "self_attn": MultiHeadAttention(
                            emb_dim, num_heads, dropout, save_attention
                        ),
                        "mlp": mlp_cls(emb_dim, feedforward_dim, dropout),
                        "layer_norm2": nn.LayerNorm(emb_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a token sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tokens with shape ``(batch, seq_len, emb_dim)``.

        Returns
        -------
        torch.Tensor
            Encoded tokens.
        """
        for layer in self.layers:
            x_norm1 = layer["layer_norm1"](x)
            attn_output = layer["self_attn"](x_norm1)
            out = x + attn_output

            x_norm2 = layer["layer_norm2"](out)
            x = out + layer["mlp"](x_norm2)
        return x

    def get_attention_maps(self) -> AttentionMaps:
        """Return saved attention maps from each encoder layer.

        Returns
        -------
        AttentionMaps
            List of attention maps or ``None`` values.
        """
        return [layer["self_attn"].last_attention for layer in self.layers]


class TremorMonoLocator(nn.Module):
    """Vision transformer for single-station tremor hypocenter prediction."""

    def __init__(
        self,
        input_length: int,
        patch_size: int,
        stride: int,
        position_emb_type: str,
        emb_dim: int,
        depth: int,
        num_heads: int,
        dropout_rate: float,
        feedforward_dim: int,
        hypo_num_output: int,
        mlp_type: str,
        input_components_number: int,
        arrival_num_output: int,
        arrival_time: bool = False,
        save_attention: bool = False,
    ) -> None:
        """Initialize the locator model.

        Parameters
        ----------
        input_length : int
            Number of waveform samples in each input.
        patch_size : int
            Patch embedding convolution kernel size.
        stride : int
            Patch embedding convolution stride.
        position_emb_type : str
            Positional embedding type.
        emb_dim : int
            Embedding dimension.
        depth : int
            Number of transformer encoder layers.
        num_heads : int
            Number of attention heads.
        dropout_rate : float
            Dropout probability.
        feedforward_dim : int
            Feed-forward hidden dimension.
        hypo_num_output : int
            Number of hypocenter output values.
        mlp_type : str
            Feed-forward block type.
        input_components_number : int
            Number of input waveform components.
        arrival_num_output : int
            Number of arrival-time output values.
        arrival_time : bool, default=False
            Whether to predict arrival time.
        save_attention : bool, default=False
            Whether attention maps should be stored.
        """
        super().__init__()
        self.hypo_num_output = hypo_num_output
        self.arrival_time = arrival_time
        num_patches = (input_length - patch_size) // stride + 1

        self.patch_embedding = ConvPatchEmbedding(
            emb_dim, patch_size, stride, input_components_number
        )

        self.hypo_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        if self.arrival_time:
            self.arrival_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        self.position_emb_type = position_emb_type
        num_special_tokens = 1 + int(self.arrival_time)

        self.positional_encoding = PositionalEncoding(
            seq_len=num_special_tokens + num_patches,
            emb_dim=emb_dim,
            position_emb_type=position_emb_type,
        )

        self.transformer = TransformerEncoder(
            emb_dim,
            num_heads,
            dropout_rate,
            feedforward_dim,
            depth,
            mlp_type,
            save_attention=save_attention,
        )
        self.hypo_mlp_heads = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, hypo_num_output),
        )
        if self.arrival_time:
            self.arrival_mlp_heads = nn.Sequential(
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, arrival_num_output),
            )

    def forward(
        self, x1: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict hypocenter and optional arrival-time parameters.

        Parameters
        ----------
        x1 : torch.Tensor
            Input waveform tensor with shape ``(batch, channels, samples)``.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            Hypocenter prediction, plus arrival-time prediction when enabled.
        """
        # x1: (batch_size, input_components_number, 6000)
        x = self.patch_embedding(x1).transpose(
            1, 2
        )  # (batch_size, num_patches, emb_dim)
        batch_size, _, _ = x.size()

        hypo_token = self.hypo_token.expand(
            batch_size, -1, -1
        )  # (batch_size, 1, emb_dim)

        tokens = [hypo_token]
        if self.arrival_time:
            arrival_token = self.arrival_token.expand(batch_size, -1, -1)
            tokens.append(arrival_token)

        x = torch.cat([*tokens, x], dim=1)

        x = self.positional_encoding(x)
        x = self.transformer(x)

        hypo_output = x[:, 0, :]
        hypo_pred = self.hypo_mlp_heads(hypo_output)
        if not self.arrival_time:
            return hypo_pred

        arrival_output = x[:, 1, :]
        arrival_pred = self.arrival_mlp_heads(arrival_output)
        return hypo_pred, arrival_pred

    def get_attention_maps(self) -> AttentionMaps:
        """Return saved attention maps from the transformer.

        Returns
        -------
        AttentionMaps
            List of attention maps or ``None`` values.
        """
        return self.transformer.get_attention_maps()


def build_vit_locator(
    cfg: ExperimentConfig,
    save_attention: bool = False,
) -> TremorMonoLocator:
    """Build a locator model from an experiment configuration.

    Parameters
    ----------
    cfg : ExperimentConfig
        Experiment configuration.
    save_attention : bool, default=False
        Whether attention maps should be stored during forward passes.

    Returns
    -------
    TremorMonoLocator
        Configured model on the training device.
    """
    c = cfg.model
    device = cfg.train.device
    in_channels = len(cfg.data.input_components)

    model = TremorMonoLocator(
        input_length=c.input_length,
        patch_size=c.patch_size,
        stride=c.stride,
        position_emb_type=c.position_emb_type,
        emb_dim=c.emb_dim,
        depth=c.depth,
        num_heads=c.num_heads,
        dropout_rate=c.dropout_rate,
        feedforward_dim=c.feedforward_dim,
        hypo_num_output=c.hypo_num_output,
        arrival_num_output=c.arrival_num_output,
        mlp_type=c.mlp_type,
        input_components_number=in_channels,
        arrival_time=cfg.data.arrival_time,
        save_attention=save_attention,
    ).to(device)

    return model


def load_vit_locator_weights(
    model: TremorMonoLocator,
    weight_path: Path,
    device: str,
    strict: bool = True,
) -> TremorMonoLocator:
    """Load saved weights into a locator model.

    Parameters
    ----------
    model : TremorMonoLocator
        Model instance to update.
    weight_path : Path
        Path to the serialized PyTorch state dictionary.
    device : str
        Device used for ``torch.load`` mapping.
    strict : bool, default=True
        Whether state-dictionary keys must match exactly.

    Returns
    -------
    TremorMonoLocator
        Model with loaded weights.
    """
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state, strict=strict)
    return model
