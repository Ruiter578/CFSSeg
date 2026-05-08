# -*- coding: utf-8 -*-
"""
Basic analytic linear modules for the analytic continual learning [1-5].

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
[5] Fang, Di, et al.
    "AIR: Analytic Imbalance Rectifier for Continual Learning."
    arXiv preprint arXiv:2408.10349 (2024).
"""

import torch
from torch.nn import functional as F
from typing import Optional, Union
from abc import abstractmethod, ABCMeta


class AnalyticLinear(torch.nn.Linear, metaclass=ABCMeta):
    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype=torch.double,
    ) -> None:
        super(torch.nn.Linear, self).__init__()  # Skip the Linear class
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gamma: float = gamma
        self.bias: bool = bias
        self.dtype = dtype

        # Linear Layer
        if bias:
            in_features += 1
        weight = torch.zeros((in_features, 0), **factory_kwargs)
        self.register_buffer("weight", weight)
    
    @torch.inference_mode()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, HW, buffersize = X.shape 
        X = X.view(B * HW, buffersize)
        X = X.to(self.weight)

        if self.bias:
            X = torch.cat((X, torch.ones(X.shape[0], 1).to(X)), dim=-1)
        P=X @ self.weight #B * HW, buffersize->B * HW, num_classes
        #B * HW, num_classes->B, H,W, num_classes
        P = P.view(B, int(HW**0.5), int(HW**0.5), -1)

        return P

    @property
    def in_features(self) -> int:
        if self.bias:
            return self.weight.shape[0] - 1
        return self.weight.shape[0]

    @property
    def out_features(self) -> int:
        return self.weight.shape[1]

    def reset_parameters(self) -> None:
        # Following the equation (4) of ACIL, self.weight is set to \hat{W}_{FCN}^{-1}
        self.weight = torch.zeros((self.weight.shape[0], 0)).to(self.weight)

    @abstractmethod
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        raise NotImplementedError()

    def update(self) -> None:
        assert torch.isfinite(self.weight).all(), (
            "Pay attention to the numerical stability! "
            "A possible solution is to increase the value of gamma. "
            "Setting self.dtype=torch.double also helps."
        )


class RecursiveLinear(AnalyticLinear):
    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype=torch.double,
    ) -> None:
        super().__init__(in_features, gamma, bias, device, dtype)
        factory_kwargs = {"device": device, "dtype": dtype}

        # Regularized Feature Autocorrelation Matrix (RFAuM)
        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R)

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """The core code of the ACIL and the G-ACIL.
        This implementation, which is different but equivalent to the equations shown in [1],
        is proposed in the G-ACIL [4], which supports mini-batch learning and the general CIL setting.
        """
        # 展平 X 和 y
        B, HW, C = X.shape
   
        X = X.view(B * HW, C)
        y = y.view(-1)
   
        # 过滤掉 y 中为 255 的样本
        mask = y != 255  # 创建掩码，过滤掉标签为 255 的样本
        X = X[mask]      # 过滤掉对应的 X
        y = y[mask]      # 过滤掉对应的 y

        X, y = X.to(self.weight), y.to(self.weight)
        num_targets = int(y.max()) + 1
        y = y.long()  # 确保 y 是整数类型
        Y = F.one_hot(y, num_classes=num_targets).to(self.weight)
        
        if self.bias:
            X = torch.cat((X, torch.ones(X.shape[0], 1).to(X)), dim=-1)

        num_targets = Y.shape[1]
        
        # print(num_targets)
        epsilon = 1e-3
        if num_targets > self.out_features:
            increment_size = num_targets - self.out_features
            tail = torch.randn((self.weight.shape[0], increment_size)).to(self.weight) * epsilon
            self.weight = torch.cat((self.weight, tail), dim=1)
        elif num_targets < self.out_features:
            increment_size = self.out_features - num_targets
            tail = torch.zeros((Y.shape[0], increment_size)).to(Y)
            Y = torch.cat((Y, tail), dim=1)
      
        # Please update your PyTorch & CUDA if the `cusolver error` occurs.
        # If you insist on using this version, doing the `torch.inverse` on CPUs might help.
        # >>> K_inv = torch.eye(X.shape[0]).to(X) + X @ self.R @ X.T
        # >>> K = torch.inverse(K_inv.cpu()).to(self.weight.device)
        # K = torch.inverse(torch.eye(X.shape[0]).to(X) + X @ self.R @ X.T)
        # # Equation (10) of ACIL
        # self.R -= self.R @ X.T @ K @ X @ self.R
        # # Equation (9) of ACIL
        # self.weight += self.R @ X.T @ (Y - X @ self.weight)
        
        # 使用Woodbury矩阵恒等式优化矩阵求逆
        # 计算 R 的逆
        R_inv = torch.inverse(self.R)
        # 计算 S = R_inv + X.T @ X
        S = R_inv + X.T @ X
        # 计算 S 的逆
        S_inv = torch.inverse(S)
        # 更新 self.R
        self.R = S_inv
        # 更新 self.weight
        self.weight += self.R @ X.T @ (Y - X @ self.weight)


class GeneralizedARM(AnalyticLinear):
    """用于广义连续增量学习（Generalized Class Incremental Learning）的解析重加权模块（ARM）。"""

    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype=torch.double,
    ) -> None:
        super().__init__(in_features, gamma, bias, device, dtype)

        self.gamma = gamma
        self.bias = bias
        self.device = device
        self.dtype = dtype

        # 初始化权重矩阵
        weight = torch.zeros((self.in_features, 0), device=device, dtype=dtype)
        self.register_buffer("weight", weight)

        # 使用字典来存储 A、C 和 cnt
        self.A_dict = {}
        self.C_dict = {}
        self.cnt_dict = {}

    @property
    def out_features(self) -> int:
        return len(self.C_dict)

    @torch.inference_mode()
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        # X: B, HW, buffer_size
        # y: B, 1, H, W
        B, HW, C = X.shape
        X = X.view(B * HW, C)   # BHW, C
        y = y.view(-1)  # y: BHW, 1
        
        # 过滤掉 y 中为 255 的样本
        mask = y != 255
        X = X[mask]
        y = y[mask]
        X = X.to(self.weight.device, dtype=self.dtype)

        # 添加偏置项（如果需要）
        if self.bias:
            bias_column = torch.ones((X.shape[0], 1), device=X.device, dtype=self.dtype)
            X = torch.cat((X, bias_column), dim=1)

        # 处理新出现的类别
        unique_classes = torch.unique(y)
        for cls in unique_classes:
            cls = int(cls.item())
            if cls not in self.C_dict:
                self.C_dict[cls] = torch.zeros((self.in_features, ), device=self.weight.device, dtype=self.dtype)
                self.A_dict[cls] = torch.zeros((self.in_features, self.in_features), device=self.weight.device, dtype=self.dtype)
                self.cnt_dict[cls] = 0

        # 累加 C 和 cnt
        for xi, yi in zip(X, y):
            cls = int(yi.item())
            self.C_dict[cls] += xi
            self.cnt_dict[cls] += 1

        # 累加 A
        for cls in unique_classes:
            cls = int(cls.item())
            mask = y == cls
            X_cls = X[mask]
            self.A_dict[cls] += X_cls.t() @ X_cls

    @torch.inference_mode()
    def update(self):
        classes = sorted(self.C_dict.keys())
        num_classes = len(classes)

        # 构建 C 矩阵和 cnt 向量
        C = torch.zeros((self.in_features, num_classes), device=self.weight.device, dtype=self.dtype)
        cnt = torch.zeros((num_classes,), device=self.weight.device, dtype=self.dtype)

        for idx, cls in enumerate(classes):
            C[:, idx] = self.C_dict[cls]
            cnt[idx] = self.cnt_dict[cls]

        # 计算 cnt 的倒数，并处理无限值
        cnt_inv = 1 / cnt
        cnt_inv[torch.isinf(cnt_inv)] = 0  # 将 inf 替换为 0
        cnt_inv_mask = cnt != 0  # 创建一个掩码，标记 cnt 中不为 0 的位置

        # 加权平均 A
        weighted_A = torch.zeros((self.in_features, self.in_features), device=self.weight.device, dtype=self.dtype)
        for idx, cls in enumerate(classes):
            if cnt_inv_mask[idx]:
                weighted_A += self.A_dict[cls] * cnt_inv[idx]

        # 添加正则化项
        A = weighted_A + self.gamma * torch.eye(self.in_features, device=self.weight.device, dtype=self.dtype)

        # 对 C 进行加权
        C = C * cnt_inv.unsqueeze(0)  # 广播机制

        # 计算权重，使用线性求解器代替矩阵求逆
        self.weight = torch.linalg.solve(A, C)