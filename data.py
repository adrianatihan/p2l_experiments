"""
Dataset loaders for P2L.

Every loader returns:
    X : np.ndarray, float32   — shape depends on dataset
    y : np.ndarray, int        — binary labels {0, 1}

MNIST:  X is (N, 784)         — flattened, StandardScaler-normalised
CIFAR-10: X is (N, 3, 32, 32) — pixel values in [0, 1]
"""

import numpy as np


# ── MNIST (10-class) ──────────────────────────────────────────────────────────

def load_mnist(n_samples=2000):
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler

    print("Loading MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, parser="auto")
    X, y = mnist.data.values, mnist.target.values.astype(np.int64)

    idx = np.random.choice(len(X), n_samples, replace=False)
    X_sub = X[idx].astype(np.float32)
    y_sub = y[idx]

    scaler = StandardScaler()
    X_sub = scaler.fit_transform(X_sub).astype(np.float32)

    return X_sub, y_sub


# ── CIFAR-10 (10-class) ───────────────────────────────────────────────────────

def load_cifar10(n_samples=2000):
    from torchvision import datasets

    print("Loading CIFAR-10 dataset...")
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True)
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True)

    all_images = np.concatenate([train_ds.data, test_ds.data], axis=0)
    all_labels = np.concatenate(
        [np.array(train_ds.targets), np.array(test_ds.targets)], axis=0
    ).astype(np.int64)

    idx = np.random.choice(len(all_images), n_samples, replace=False)
    X = all_images[idx].astype(np.float32) / 255.0
    X = X.transpose(0, 3, 1, 2)  # NCHW
    y = all_labels[idx]

    return X, y


# ── CIFAR-100 (100-class) ─────────────────────────────────────────────────────

def load_cifar100(n_samples=2000):
    from torchvision import datasets

    print("Loading CIFAR-100 dataset...")
    train_ds = datasets.CIFAR100(root="./data", train=True, download=True)
    test_ds = datasets.CIFAR100(root="./data", train=False, download=True)

    all_images = np.concatenate([train_ds.data, test_ds.data], axis=0)
    all_labels = np.concatenate(
        [np.array(train_ds.targets), np.array(test_ds.targets)], axis=0
    ).astype(np.int64)

    idx = np.random.choice(len(all_images), n_samples, replace=False)
    X = all_images[idx].astype(np.float32) / 255.0
    X = X.transpose(0, 3, 1, 2)  # NCHW
    y = all_labels[idx]

    return X, y


# ── Registry ──────────────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    "mnist":    (load_mnist,    10),
    "cifar10":  (load_cifar10,  10),
    "cifar100": (load_cifar100, 100),
}


def load_dataset(name: str, n_samples: int = 2000):
    """Load a dataset by name. Returns (X, y, num_classes)."""
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    loader, num_classes = DATASET_REGISTRY[name]
    X, y = loader(n_samples=n_samples)
    return X, y, num_classes