"""
Broad CIFAR-10 Modal example for Hyppo.

This example deliberately exposes many architectural, optimization, and
augmentation decisions as hyperparameters so Hyppo has a wide search surface.
"""

import modal

app = modal.App("hpo-agent")
DEFAULT_RUN_TIME_MINUTES = 20
MODAL_HARD_TIMEOUT_MINUTES = 90
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "torchvision",
    "wandb",
    "numpy",
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("wandb-secret")],
    gpu="T4",
    timeout=MODAL_HARD_TIMEOUT_MINUTES * 60,
)
def train_model(
    learning_rate: float = 3e-3,
    batch_size: int = 128,
    max_epochs: int = 20,
    num_stages: int = 3,
    blocks_per_stage: int = 2,
    base_channels: int = 64,
    channel_multiplier: float = 2.0,
    kernel_size: int = 3,
    block_style: str = "double",
    use_depthwise_separable: bool = False,
    use_residual: bool = True,
    norm_type: str = "batch",
    group_norm_groups: int = 8,
    activation: str = "gelu",
    activation_negative_slope: float = 0.1,
    pool_type: str = "max",
    global_pool: str = "avg",
    conv_dropout: float = 0.05,
    classifier_dropout: float = 0.2,
    head_hidden_dim: int = 256,
    optimizer_name: str = "adamw",
    momentum: float = 0.9,
    nesterov: bool = True,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    gradient_clip_norm: float = 1.0,
    lr_schedule: str = "cosine",
    warmup_steps: int = 100,
    mixup_alpha: float = 0.0,
    augment_policy: str = "standard",
    random_crop_padding: int = 4,
    horizontal_flip_prob: float = 0.5,
    random_erasing_prob: float = 0.0,
    data_loader_workers: int = 2,
    seed: int = 1337,
    max_time_minutes: int = DEFAULT_RUN_TIME_MINUTES,
    wandb_project: str = "hpo-agent",
    wandb_entity: str | None = None,
    run_name: str | None = None,
) -> dict:
    import random
    import time

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as T
    import wandb

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    effective_max_time_minutes = max(1, min(int(max_time_minutes), MODAL_HARD_TIMEOUT_MINUTES))
    if effective_max_time_minutes != int(max_time_minutes):
        print(
            "Requested max_time_minutes="
            f"{max_time_minutes} exceeds bounds; using {effective_max_time_minutes}."
        )

    def _round_channels(value: float) -> int:
        return max(8, int(round(value / 8.0) * 8))

    def _make_activation() -> nn.Module:
        if activation == "relu":
            return nn.ReLU(inplace=True)
        if activation == "silu":
            return nn.SiLU(inplace=True)
        if activation == "leaky_relu":
            return nn.LeakyReLU(negative_slope=activation_negative_slope, inplace=True)
        return nn.GELU()

    def _make_norm(channels: int) -> nn.Module:
        if norm_type == "group":
            groups = max(1, min(group_norm_groups, channels))
            while channels % groups != 0 and groups > 1:
                groups -= 1
            return nn.GroupNorm(groups, channels)
        if norm_type == "instance":
            return nn.InstanceNorm2d(channels, affine=True)
        if norm_type == "none":
            return nn.Identity()
        return nn.BatchNorm2d(channels)

    def _make_pool() -> nn.Module:
        if pool_type == "avg":
            return nn.AvgPool2d(kernel_size=2)
        return nn.MaxPool2d(kernel_size=2)

    def _make_global_pool() -> nn.Module:
        if global_pool == "max":
            return nn.AdaptiveMaxPool2d(1)
        return nn.AdaptiveAvgPool2d(1)

    def _conv_layers(in_channels: int, out_channels: int) -> list[nn.Module]:
        padding = kernel_size // 2
        if use_depthwise_separable:
            return [
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    groups=in_channels,
                    bias=False,
                ),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            ]
        return [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        ]

    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            layers = []
            stages = 2 if block_style == "double" else 1
            current_in = in_channels
            for _ in range(stages):
                layers.extend(_conv_layers(current_in, out_channels))
                layers.append(_make_norm(out_channels))
                layers.append(_make_activation())
                if conv_dropout > 0:
                    layers.append(nn.Dropout2d(conv_dropout))
                current_in = out_channels
            self.net = nn.Sequential(*layers)
            self.use_residual = use_residual
            if in_channels == out_channels:
                self.skip = nn.Identity()
            else:
                self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        def forward(self, x):
            out = self.net(x)
            if self.use_residual:
                out = out + self.skip(x)
            return out

    class FlexibleCifarNet(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            in_channels = base_channels
            current_channels = base_channels

            stem = [
                nn.Conv2d(3, base_channels, kernel_size=3, padding=1, bias=False),
                _make_norm(base_channels),
                _make_activation(),
            ]
            if conv_dropout > 0:
                stem.append(nn.Dropout2d(conv_dropout))
            layers.extend(stem)

            for stage_idx in range(num_stages):
                out_channels = _round_channels(
                    current_channels * (channel_multiplier ** stage_idx)
                )
                for block_idx in range(blocks_per_stage):
                    block_in = in_channels if block_idx == 0 else out_channels
                    layers.append(ConvBlock(block_in, out_channels))
                    in_channels = out_channels
                if stage_idx < num_stages - 1:
                    layers.append(_make_pool())

            self.features = nn.Sequential(*layers)
            self.global_pool = _make_global_pool()
            head_layers = [nn.Flatten()]
            if head_hidden_dim > 0:
                head_layers.extend(
                    [
                        nn.Linear(in_channels, head_hidden_dim),
                        _make_activation(),
                        nn.Dropout(classifier_dropout),
                        nn.Linear(head_hidden_dim, 10),
                    ]
                )
            else:
                head_layers.extend(
                    [
                        nn.Dropout(classifier_dropout),
                        nn.Linear(in_channels, 10),
                    ]
                )
            self.classifier = nn.Sequential(*head_layers)

        def forward(self, x):
            x = self.features(x)
            x = self.global_pool(x)
            return self.classifier(x)

    def _build_transforms():
        train_transforms = []
        if random_crop_padding > 0:
            train_transforms.append(T.RandomCrop(32, padding=random_crop_padding))
        if horizontal_flip_prob > 0:
            train_transforms.append(T.RandomHorizontalFlip(p=horizontal_flip_prob))
        if augment_policy == "autoaugment":
            train_transforms.append(T.AutoAugment(T.AutoAugmentPolicy.CIFAR10))
        elif augment_policy == "trivialaugment":
            train_transforms.append(T.TrivialAugmentWide())
        train_transforms.extend([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
        if random_erasing_prob > 0:
            train_transforms.append(
                T.RandomErasing(
                    p=random_erasing_prob,
                    scale=(0.02, 0.2),
                    ratio=(0.3, 3.3),
                    value="random",
                )
            )

        val_transforms = [T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)]
        return T.Compose(train_transforms), T.Compose(val_transforms)

    def _build_optimizer(model: nn.Module):
        if optimizer_name == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=weight_decay,
            )
        if optimizer_name == "rmsprop":
            return torch.optim.RMSprop(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )

    def _build_scheduler(optimizer, total_steps: int):
        if total_steps <= 1 or lr_schedule == "constant":
            return None
        if lr_schedule == "onecycle":
            pct_start = 0.1
            if warmup_steps > 0:
                pct_start = min(0.45, max(0.05, warmup_steps / total_steps))
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                total_steps=total_steps,
                pct_start=pct_start,
            )

        if lr_schedule == "step":
            main = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(1, total_steps // 3),
                gamma=0.3,
            )
        else:
            main = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
            )

        if warmup_steps <= 0:
            return main

        effective_warmup = min(warmup_steps, max(1, total_steps - 1))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=effective_warmup,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, main],
            milestones=[effective_warmup],
        )

    def _apply_mixup(images, labels):
        if mixup_alpha <= 0:
            return images, labels, None, 1.0
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        index = torch.randperm(images.size(0), device=images.device)
        mixed_images = lam * images + (1.0 - lam) * images[index]
        return mixed_images, labels, labels[index], lam

    train_transform, val_transform = _build_transforms()
    train_ds = torchvision.datasets.CIFAR10(
        root="/tmp/cifar10",
        train=True,
        download=True,
        transform=train_transform,
    )
    val_ds = torchvision.datasets.CIFAR10(
        root="/tmp/cifar10",
        train=False,
        download=True,
        transform=val_transform,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": data_loader_workers,
        "pin_memory": True,
    }
    if data_loader_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlexibleCifarNet().to(device)
    optimizer = _build_optimizer(model)
    total_steps = max(1, len(train_loader) * max_epochs)
    scheduler = _build_scheduler(optimizer, total_steps)
    parameter_count = sum(param.numel() for param in model.parameters())

    config = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "num_stages": num_stages,
        "blocks_per_stage": blocks_per_stage,
        "base_channels": base_channels,
        "channel_multiplier": channel_multiplier,
        "kernel_size": kernel_size,
        "block_style": block_style,
        "use_depthwise_separable": use_depthwise_separable,
        "use_residual": use_residual,
        "norm_type": norm_type,
        "group_norm_groups": group_norm_groups,
        "activation": activation,
        "activation_negative_slope": activation_negative_slope,
        "pool_type": pool_type,
        "global_pool": global_pool,
        "conv_dropout": conv_dropout,
        "classifier_dropout": classifier_dropout,
        "head_hidden_dim": head_hidden_dim,
        "optimizer_name": optimizer_name,
        "momentum": momentum,
        "nesterov": nesterov,
        "beta1": beta1,
        "beta2": beta2,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "gradient_clip_norm": gradient_clip_norm,
        "lr_schedule": lr_schedule,
        "warmup_steps": warmup_steps,
        "mixup_alpha": mixup_alpha,
        "augment_policy": augment_policy,
        "random_crop_padding": random_crop_padding,
        "horizontal_flip_prob": horizontal_flip_prob,
        "random_erasing_prob": random_erasing_prob,
        "data_loader_workers": data_loader_workers,
        "seed": seed,
        "max_time_minutes": effective_max_time_minutes,
        "modal_hard_timeout_minutes": MODAL_HARD_TIMEOUT_MINUTES,
        "parameter_count": parameter_count,
    }
    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        id=run_name,
        name=run_name,
        resume="allow",
        settings=wandb.Settings(init_timeout=180),
        config=config,
    )
    run_id = run.id

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    best_val_loss = float("inf")
    best_val_loss_time_seconds = 0.0
    best_val_loss_progress_percent = 0.0
    best_accuracy = 0.0
    best_accuracy_time_seconds = 0.0
    best_accuracy_progress_percent = 0.0
    best_val_loss_at_best_accuracy = float("inf")
    avg_val_loss = float("inf")
    accuracy = 0.0
    start_time = time.monotonic()
    max_time_seconds = effective_max_time_minutes * 60

    for epoch in range(max_epochs):
        model.train()
        train_losses = []

        for images, labels in train_loader:
            elapsed_time_seconds = time.monotonic() - start_time
            if elapsed_time_seconds >= max_time_seconds:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images, targets_a, targets_b, lam = _apply_mixup(images, labels)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if targets_b is None:
                loss = criterion(logits, targets_a)
            else:
                loss = lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)
            loss.backward()

            if gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            train_losses.append(loss.item())

        if not train_losses:
            break

        avg_train_loss = float(np.mean(train_losses))

        model.eval()
        val_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, labels)
                val_losses.append(loss.item())
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = float(np.mean(val_losses))
        accuracy = correct / total if total > 0 else 0.0
        elapsed_time_seconds = time.monotonic() - start_time
        progress_percent = min((elapsed_time_seconds / max_time_seconds) * 100.0, 100.0)
        current_lr = optimizer.param_groups[0]["lr"]

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_loss_time_seconds = elapsed_time_seconds
            best_val_loss_progress_percent = progress_percent

        if (
            accuracy > best_accuracy
            or (
                accuracy == best_accuracy
                and avg_val_loss < best_val_loss_at_best_accuracy
            )
        ):
            best_accuracy = accuracy
            best_accuracy_time_seconds = elapsed_time_seconds
            best_accuracy_progress_percent = progress_percent
            best_val_loss_at_best_accuracy = avg_val_loss

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "accuracy": accuracy,
                "elapsed_time_seconds": elapsed_time_seconds,
                "progress_percent": progress_percent,
            }
        )

        print(
            f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}, acc={accuracy:.4f}, "
            f"best_acc={best_accuracy:.4f}, "
            f"lr={current_lr:.6f}, params={parameter_count}, "
            f"time={elapsed_time_seconds:.1f}s, progress={progress_percent:.1f}%"
        )

        if elapsed_time_seconds >= max_time_seconds:
            break

    wandb.finish()

    return {
        "run_id": run_id,
        "best_val_loss": float(best_val_loss),
        "best_time_seconds": float(best_val_loss_time_seconds),
        "best_progress_percent": float(best_val_loss_progress_percent),
        "final_val_loss": float(avg_val_loss),
        "final_accuracy": float(accuracy),
        "best_accuracy": float(best_accuracy),
        "best_accuracy_time_seconds": float(best_accuracy_time_seconds),
        "best_accuracy_progress_percent": float(best_accuracy_progress_percent),
        "best_val_loss_at_best_accuracy": float(best_val_loss_at_best_accuracy),
        "parameter_count": int(parameter_count),
    }
