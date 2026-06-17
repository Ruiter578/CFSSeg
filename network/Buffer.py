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
        rhl_seed: int = -1,
        rhl_init: str = "gaussian",
        rhl_scale_mode: str = "legacy",
    ) -> None:
        super(torch.nn.Linear, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.rhl_norm = rhl_norm
        self.rhl_norm_eps = rhl_norm_eps
        self.rhl_seed = rhl_seed
        self.rhl_init = rhl_init
        self.rhl_scale_mode = rhl_scale_mode
        self.activation: activation_t = (
            torch.nn.Identity() if activation is None else activation
        )

        W = torch.empty((out_features, in_features), **factory_kwargs)
        b = torch.empty(out_features, **factory_kwargs) if bias else None

        # 使用 buffer 而不是 Parameter：RHL 是固定随机映射，不参与反向传播训练。
        self.register_buffer("weight", W)
        self.register_buffer("bias", b)

        # 随机初始化：
        # - rhl_seed < 0：保持原始代码行为，使用当前全局 RNG 状态初始化 RHL；
        # - rhl_seed >= 0：只重置 RandomBuffer/RHL 的随机映射，形成 RHL-SE 的不同成员；
        # - fork_rng 会在初始化后恢复外部 RNG 状态，因此不会改变 DataLoader 顺序、
        #   数据增强、RecursiveLinear 初始化等其他随机过程。
        self._reset_parameters_with_optional_seed(rhl_seed)

    def _reset_parameters_with_optional_seed(self, rhl_seed: int) -> None:
        # -1 是兼容开关：不隔离 RHL 独立种子，沿用当前全局 RNG 状态。
        if rhl_seed is None or int(rhl_seed) < 0:
            self._reset_rhl_parameters()
            return

        seed = int(rhl_seed)
        cuda_devices = []
        if self.weight.is_cuda:
            cuda_devices = [self.weight.device.index or 0]

        # 只在这个上下文里临时切换随机种子。退出后 PyTorch 会恢复进入前的 RNG 状态，
        # 这是区分“只改 RHL 随机映射”和“改全局 random_seed”的关键实现。
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if self.weight.is_cuda:
                torch.cuda.manual_seed_all(seed)
            self._reset_rhl_parameters()

    def _reset_rhl_parameters(self) -> None:
        if self.rhl_init == "gaussian":
            self._init_gaussian()
        elif self.rhl_init == "orthogonal":
            self._init_orthogonal(antithetic=False)
        elif self.rhl_init == "orthogonal_antithetic":
            self._init_orthogonal(antithetic=True)
        else:
            raise ValueError(f"Unknown rhl_init: {self.rhl_init}")

    def _init_gaussian(self) -> None:
        if self.rhl_scale_mode == "legacy":
            # Preserve the exact PyTorch Linear initialization path used by the
            # existing baseline. The docs call it gaussian RHL, but the code path
            # is Linear.reset_parameters(); keeping it avoids baseline drift.
            self.reset_parameters()
            return

        with torch.no_grad():
            if self.rhl_scale_mode == "kaiming":
                self.weight.normal_(mean=0.0, std=math.sqrt(2.0 / self.in_features))
            elif self.rhl_scale_mode == "unit":
                self.weight.normal_(mean=0.0, std=1.0)
                self.weight.copy_(F.normalize(self.weight, p=2, dim=1))
            else:
                raise ValueError(f"Unknown rhl_scale_mode: {self.rhl_scale_mode}")
            if self.bias is not None:
                self.bias.zero_()

    def _scale_for_mode(self) -> float:
        if self.rhl_scale_mode == "legacy":
            # Linear.reset_parameters() gives each row expected norm 1/sqrt(3).
            return 1.0 / math.sqrt(3.0)
        if self.rhl_scale_mode == "kaiming":
            # Orthogonal rows have norm 1. Multiplying by sqrt(2) gives per-entry
            # std close to sqrt(2 / in_features), matching ReLU/Kaiming scale.
            return math.sqrt(2.0)
        if self.rhl_scale_mode == "unit":
            return 1.0
        raise ValueError(f"Unknown rhl_scale_mode: {self.rhl_scale_mode}")

    def _init_orthogonal_blocks(self, rows, cols, device, dtype) -> torch.Tensor:
        blocks = []
        remain = rows
        while remain > 0:
            q, _ = torch.linalg.qr(torch.randn(cols, cols, device=device, dtype=dtype))
            block = q[: min(cols, remain)]
            blocks.append(block)
            remain -= block.shape[0]
        return torch.cat(blocks, dim=0)

    def _init_orthogonal(self, antithetic: bool) -> None:
        device = self.weight.device
        dtype = self.weight.dtype
        rows = math.ceil(self.out_features / 2) if antithetic else self.out_features
        base = self._init_orthogonal_blocks(rows, self.in_features, device, dtype)
        if antithetic:
            weight = torch.cat([base, -base], dim=0)[: self.out_features]
        else:
            weight = base

        with torch.no_grad():
            self.weight.copy_(weight * self._scale_for_mode())
            if self.bias is not None:
                self.bias.zero_()

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [B, H*W, C]。RHL 先执行 CFSSeg 原论文中的固定随机投影和非线性激活，
        # 然后可选地对每个像素的高维随机特征做归一化，再交给 C-RLS 闭式分类器。
        X = X.to(self.weight)
        Z = self.activation(super().forward(X))

        # 兼容旧 AIR checkpoint：旧模型里没有 rhl_norm / rhl_norm_eps 属性，
        # 用 getattr 默认值可以让旧权重仍按 baseline RHL 逻辑推理。
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
