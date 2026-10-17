"""Independent FP64 storage reference with explicit FP32 rounding bounds.

Author: Winston Cai. Quantization is discontinuous: comparing two independently
reduced FP32 RMSNorm results after FP8 rounding with a generic relative tolerance
can reject a valid adjacent code at a BF16 midpoint. We instead bound the FP32
normalization error before the prescribed BF16/FP8 rounding, then verify the
actual scale and value against that interval. Attention tolerances are separate.
"""

from __future__ import annotations
import torch


def check_fp8_storage(
    decoded: torch.Tensor,
    codes: torch.Tensor,
    latent: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    name: str,
    keep_mask: torch.Tensor | None = None,
) -> None:
    """Check FP8 coordinates [N, 448|512] and UE8M0 scale codes [N, 7|8].

    Raw latent [N,512] and weights [512] are finite FP32; an optional bool mask
    selects raw coordinates before normalization. No producer kernel is used.
    The conservative gamma bound covers 512 nonnegative products/sums plus
    normalization, approximate reciprocal square root and weight multiplication.
    It is applied before quantization, never as a tolerance on stored FP8 codes.
    """
    if (
        latent.dtype != torch.float32
        or weight.dtype != torch.float32
        or latent.shape[1] != 512
    ):
        raise ValueError(f"{name}: finite FP32 D512 storage inputs required")
    if (
        not bool(torch.isfinite(latent).all() & torch.isfinite(weight).all())
        or eps <= 0
    ):
        raise ValueError(f"{name}: nonfinite inputs or invalid epsilon")
    if decoded.shape[1] not in (448, 512) or codes.shape != (
        len(latent),
        decoded.shape[1] // 64,
    ):
        raise ValueError(f"{name}: invalid decoded/scale geometry")
    x = latent.double()
    if keep_mask is not None:
        if keep_mask.dtype != torch.bool or keep_mask.shape != latent.shape:
            raise ValueError(f"{name}: invalid raw selection mask")
        x = x.masked_fill(~keep_mask, 0)
    normalized = (
        x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.double()
    )
    # IEEE FP32 unit roundoff. 2*gamma_520 also bounds rsqrt error and the
    # perturbation of the denominator; our finite Gaussian fixtures are normal.
    u = 2.0**-24
    gamma = 520 * u / (1 - 520 * u)
    error = normalized.abs() * (2 * gamma)
    low = (normalized - error).to(torch.bfloat16).float()[:, : decoded.shape[1]]
    high = (normalized + error).to(torch.bfloat16).float()[:, : decoded.shape[1]]
    lo, hi = low.reshape(len(x), -1, 64), high.reshape(len(x), -1, 64)
    minimum_abs = torch.where(
        (lo <= 0) & (hi >= 0), 0, torch.minimum(lo.abs(), hi.abs())
    )
    maximum_abs = torch.maximum(lo.abs(), hi.abs())
    lower_code = (
        torch.ceil(torch.log2(minimum_abs.amax(-1).clamp_min(1e-4) / 448)) + 127
    )
    upper_code = (
        torch.ceil(torch.log2(maximum_abs.amax(-1).clamp_min(1e-4) / 448)) + 127
    )
    scale_ok = (codes.float() >= lower_code) & (codes.float() <= upper_code)
    if not bool(scale_ok.all()):
        raise AssertionError(
            f"{name}: UE8M0 scale outside independent normalization bounds"
        )
    scale = torch.exp2(codes.float() - 127).repeat_interleave(64, -1)
    lower_value = (low / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
    upper_value = (high / scale).clamp(-448, 448).to(
        torch.float8_e4m3fn
    ).float() * scale
    valid = (
        (decoded.float() >= lower_value)
        & (decoded.float() <= upper_value)
        & torch.isfinite(decoded)
    )
    if not bool(valid.all()):
        first = tuple((~valid).nonzero()[0].tolist())
        raise AssertionError(
            f"{name}: stored FP8 value outside independent quantization interval at {first}: "
            f"{float(decoded[first])} outside [{float(lower_value[first])}, {float(upper_value[first])}]"
        )
