# -*- coding: utf-8 -*-
"""
Buffer layers for the analytic learning based CIL [1-4].

References:
[1] Zhuang, Huiping, et al.
    "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
    Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
[2] Zhuang, Huiping, et al.
    "GKEAL: Gaussian Kernel Embedded Analytic Learning for Few-Shot Class Incremental Task."
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
[3] Zhuang, Huiping, et al.
    "DS-AL: A Dual-Stream Analytic Learning for Exemplar-Free Class-Incremental Learning."
    Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.
[4] Zhuang, Huiping, et al.
    "G-ACIL: Analytic Learning for Exemplar-Free Generalized Class Incremental Learning"
    arXiv preprint arXiv:2403.15706 (2024).
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Union, Callable
from abc import ABCMeta, abstractmethod


activation_t = Union[Callable[[torch.Tensor], torch.Tensor], torch.nn.Module]


class Buffer(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class RandomBuffer(torch.nn.Linear, Buffer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=torch.float,
        activation: Optional[activation_t] = torch.relu_,
        rhl_norm: str = "none",
        rhl_norm_eps: float = 1e-6,
    ) -> None:
        super(torch.nn.Linear, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.rhl_norm = rhl_norm
        self.rhl_norm_eps = rhl_norm_eps
        self.activation: activation_t = (
            torch.nn.Identity() if activation is None else activation
        )

        W = torch.empty((out_features, in_features), **factory_kwargs)
        b = torch.empty(out_features, **factory_kwargs) if bias else None

        # Using buffer instead of parameter
        self.register_buffer("weight", W)
        self.register_buffer("bias", b)

        # Random Initialization
        self.reset_parameters()

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [B, H*W, C]. RHL first applies the fixed random projection and
        # non-linearity from CFSSeg, then optionally normalizes each pixel's
        # high-dimensional feature vector before C-RLS consumes it.
        X = X.to(self.weight)
        Z = self.activation(super().forward(X))

        # Old AIR checkpoints were saved before these attributes existed.  Use
        # getattr defaults so those pickled models still run with baseline RHL.
        norm = getattr(self, "rhl_norm", "none")
        eps = getattr(self, "rhl_norm_eps", 1e-6)

        if norm == "none":
            return Z
        if norm == "l2":
            # Pure row-wise L2 normalization: every pixel feature has unit norm.
            return F.normalize(Z, p=2, dim=-1, eps=eps)
        if norm == "l2_sqrt":
            # Preserve the feature direction and keep total row energy near the
            # original high-dimensional scale, which makes gamma=1 less distorted
            # than pure L2 when buffer is large.
            return F.normalize(Z, p=2, dim=-1, eps=eps) * math.sqrt(Z.shape[-1])
        if norm == "layernorm":
            # No affine parameters: this is still a fixed feature transform, not
            # a trainable normalization layer that would change the closed-form story.
            return F.layer_norm(Z, (Z.shape[-1],), weight=None, bias=None, eps=eps)
        raise ValueError(f"Unknown rhl_norm: {norm}")


class GaussianKernel(Buffer):
    def __init__(
        self, mean: torch.Tensor, sigma: float = 1, device=None, dtype=torch.float
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        factory_kwargs = {"device": device, "dtype": dtype}
        assert len(mean.shape) == 2, "The mean should be a 2D tensor."
        mean = mean[None, :, :].to(**factory_kwargs)
        beta = 1 / (2 * (sigma**2))
        self.register_buffer("mean", mean)
        self.register_buffer("beta", torch.tensor(beta, **factory_kwargs))

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.square_(torch.cdist(X.to(self.mean), self.mean))
        return torch.exp_(X.mul_(-self.beta))

    def init(self, X: torch.Tensor, size: Optional[int] = None) -> None:
        if size is not None:
            if size <= X.shape[0]:
                idx = torch.randperm(size).to(X.device)
                X = X[idx]
            else:
                # The buffer size is suggested to be greater than the number of initial samples.
                # Generate center vectors randomly
                n_require = size - X.shape[0]
                W_proj = torch.normal(mean=0, std=1, size=(n_require, X.shape[0])).to(X)
                W_proj /= torch.sum(W_proj, dim=0)
                X = torch.cat([X, W_proj @ X], dim=0)
        self.mean = X.to(self.mean)
