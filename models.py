"""
Model architectures for P2L.

All models output a single raw logit (shape (N, 1)) for binary classification.
The model_fn returned by get_model_fn() is a zero-argument callable that
produces a fresh, randomly initialised model each time it is called.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
#  MNIST — Fully connected MLP
# ═══════════════════════════════════════════════════════════════════════════════

class BinaryMLP(nn.Module):
    """
    784 → 600 → 600 → 600 → 1   (no dropout, clean for verification)
    """

    def __init__(self, input_size=784, hidden_size=600):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)


# ═══════════════════════════════════════════════════════════════════════════════
#  CIFAR-10 — ResNet (no BatchNorm, skip connections, auto_LiRPA-friendly)
# ═══════════════════════════════════════════════════════════════════════════════

class BasicBlock(nn.Module):
    """ResNet basic block with skip connection (no BN)."""

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, 3, stride=stride, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return F.relu(out)


class CifarResNet(nn.Module):
    """
    Lightweight ResNet for CIFAR-10 binary classification.
    conv1(3→64) → 4 layers of BasicBlocks → pool_conv(4×4→1×1) → fc(512→1)
    No BatchNorm, no AvgPool, no conv bias — clean for ONNX/onnx2pytorch.
    Pool conv replaces avg_pool2d to avoid onnx2pytorch count_include_pad bug.
    """
 
    def __init__(self):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(64, 1, stride=1)
        self.layer2 = self._make_layer(128, 1, stride=2)
        self.layer3 = self._make_layer(256, 1, stride=2)
        self.layer4 = self._make_layer(512, 1, stride=2)
        # Learned pooling: 512×4×4 → 512×1×1 (replaces avg_pool2d)
        self.pool_conv = nn.Conv2d(512, 512, kernel_size=4, stride=4, bias=False)
        self.fc = nn.Linear(512, 1)
        self._init_weights()
 
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)
 
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
 
    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 3, 32, 32)
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool_conv(out)       # 512×4×4 → 512×1×1
        out = torch.flatten(out, 1)     # 512
        return self.fc(out)

class CifarCNN(nn.Module):
    """
    Plain CNN for CIFAR-10 binary classification.
 
    conv(3→16,s1) → relu → conv(16→16,s2) → relu →
    conv(16→32,s1) → relu → conv(32→32,s2) → relu →
    flatten(32×8×8=2048) → fc(2048→128) → relu → fc(128→1)
 
    No skip connections, no BN, no conv bias — clean and fast
    for CROWN/BaB bound propagation.
    """
 
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)
        self._init_weights()
 
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
 
    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 3, 32, 32)
        out = F.relu(self.conv1(x))    # 16×32×32
        out = F.relu(self.conv2(out))  # 16×16×16
        out = F.relu(self.conv3(out))  # 32×16×16
        out = F.relu(self.conv4(out))  # 32×8×8
        out = torch.flatten(out, 1)    # 2048
        out = F.relu(self.fc1(out))    # 128
        return self.fc2(out)           # 1

# ═══════════════════════════════════════════════════════════════════════════════
#  Registry
# ═══════════════════════════════════════════════════════════════════════════════

# Maps name → (model_class, input_shape_including_batch_dim)
# input_shape is used for verification dummy inputs.
MODEL_REGISTRY = {
    "mnist_mlp": (BinaryMLP, (1, 784)),
    "cifar_resnet": (CifarResNet, (1, 3, 32, 32)),
}


def get_model_fn(name: str):
    """
    Return (model_fn, input_shape).

    model_fn() → fresh nn.Module   (call it each time you need a new model)
    input_shape → tuple like (1, 3, 32, 32)
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    cls, shape = MODEL_REGISTRY[name]
    return cls, shape
