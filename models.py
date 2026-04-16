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

class MLP(nn.Module):
    """784 → 600 → 600 → 600 → num_classes"""

    def __init__(self, num_classes=10, input_size=784, hidden_size=600):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_classes)
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
#  CIFAR-10 — Plain CNN (verification-friendly)
# ═══════════════════════════════════════════════════════════════════════════════

class CifarCNN(nn.Module):
    """
    conv(3→16,s1) → relu → conv(16→16,s2) → relu →
    conv(16→32,s1) → relu → conv(32→32,s2) → relu →
    flatten(32×8×8=2048) → fc(2048→128) → relu → fc(128→num_classes)
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
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
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        return self.fc2(out)


# ═══════════════════════════════════════════════════════════════════════════════
#  CIFAR-10 — ResNet
# ═══════════════════════════════════════════════════════════════════════════════

class BasicBlock(nn.Module):
    """ResNet basic block (no BN, no conv bias). ReLU after Add."""

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1,
                               padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return F.relu(out)


class CifarResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(16, 1, stride=1)
        self.layer2 = self._make_layer(32, 1, stride=2)
        self.layer3 = self._make_layer(64, 1, stride=2)
        self.layer4 = self._make_layer(128, 1, stride=2)
        self.pool_conv = nn.Conv2d(128, 128, kernel_size=4, stride=4,
                                   bias=False)
        self.fc = nn.Linear(128, num_classes)
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
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
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
        out = self.pool_conv(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


# ═══════════════════════════════════════════════════════════════════════════════
#  CIFAR-100 — VNN-COMP 2025 ResNet benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

class BNBasicBlock(nn.Module):
    """ResNet basic block with BatchNorm. No ReLU after the residual Add."""

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return out


def _init_cifar100_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                    nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def _make_bn_layer(module, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for s in strides:
        layers.append(BNBasicBlock(module.in_planes, planes, s))
        module.in_planes = planes
    return nn.Sequential(*layers)


class Cifar100ResNetMedium(nn.Module):
    """VNN-COMP CIFAR100_resnet_medium (2.54M params, 19 Conv + 2 FC)."""

    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.in_planes = 64
        self.layer1 = _make_bn_layer(self, 128, 4, stride=2)
        self.layer2 = _make_bn_layer(self, 128, 4, stride=2)

        self.fc1 = nn.Linear(128 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, num_classes)
        _init_cifar100_weights(self)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 3, 32, 32)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        return self.fc2(out)


class Cifar100ResNetLarge(nn.Module):
    """VNN-COMP CIFAR100_resnet_large (3.81M params, 20 Conv + 2 FC)."""

    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.in_planes = 64
        self.layer1 = _make_bn_layer(self, 64,  2, stride=1)
        self.layer2 = _make_bn_layer(self, 128, 2, stride=2)
        self.layer3 = _make_bn_layer(self, 128, 2, stride=2)
        self.layer4 = _make_bn_layer(self, 256, 2, stride=2)

        self.fc1 = nn.Linear(256 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, num_classes)
        _init_cifar100_weights(self)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 3, 32, 32)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        return self.fc2(out)


# ═══════════════════════════════════════════════════════════════════════════════
#  Registry
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "mnist_mlp":     (MLP,          (1, 784),        10),
    "cifar_cnn":     (CifarCNN,     (1, 3, 32, 32),  10),
    "cifar_resnet":  (CifarResNet,  (1, 3, 32, 32),  10),
    "cifar100_resnet_medium": (Cifar100ResNetMedium, (1, 3, 32, 32), 100),
    "cifar100_resnet_large":  (Cifar100ResNetLarge,  (1, 3, 32, 32), 100),
}


def get_model_fn(name: str, num_classes: int = None):
    """
    Returns (model_factory, input_shape) where model_factory() builds a
    fresh model with the correct num_classes for the task.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    cls, shape, default_nc = MODEL_REGISTRY[name]
    nc = num_classes if num_classes is not None else default_nc
    return (lambda: cls(num_classes=nc)), shape