"""
Example Modal training function: small CNN for CIFAR-10 image classification.
Accepts hyperparameters as arguments, logs to W&B.
Trains fast (~2-3 min on a T4/A10G).
"""

import modal

app = modal.App("hpo-agent")
RUN_TIMEOUT_MINUTES = 20

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
    timeout=RUN_TIMEOUT_MINUTES * 60,
)
def train_model(
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    num_conv_layers: int = 3,
    num_filters: int = 64,
    dropout: float = 0.2,
    weight_decay: float = 1e-4,
    warmup_steps: int = 100,
    label_smoothing: float = 0.0,
    gradient_clip_norm: float = 1.0,
    lr_schedule: str = "cosine",
    use_batch_norm: bool = True,
    wandb_project: str = "hpo-agent",
    wandb_entity: str | None = None,
    run_name: str | None = None,
) -> dict:
    import numpy as np
    import time
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as T
    import wandb

    # --- W&B init ---
    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        id=run_name,
        name=run_name,
        resume="allow",
        settings=wandb.Settings(init_timeout=180),
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_conv_layers": num_conv_layers,
            "num_filters": num_filters,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps,
            "label_smoothing": label_smoothing,
            "gradient_clip_norm": gradient_clip_norm,
            "lr_schedule": lr_schedule,
            "use_batch_norm": use_batch_norm,
            "max_time_minutes": RUN_TIMEOUT_MINUTES,
        },
    )
    run_id = run.id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data ---
    transform_train = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    transform_val = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_ds = torchvision.datasets.CIFAR10(
        root="/tmp/cifar10", train=True, download=True, transform=transform_train
    )
    val_ds = torchvision.datasets.CIFAR10(
        root="/tmp/cifar10", train=False, download=True, transform=transform_val
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # --- Model: simple CNN ---
    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            in_channels = 3
            for i in range(num_conv_layers):
                out_channels = num_filters * (2 ** min(i, 2))  # 64, 128, 256, 256, ...
                layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
                if use_batch_norm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.MaxPool2d(2))
                layers.append(nn.Dropout2d(dropout))
                in_channels = out_channels

            self.features = nn.Sequential(*layers)

            # Compute flattened size: 32x32 halved num_conv_layers times
            spatial = 32 // (2**num_conv_layers)
            if spatial < 1:
                spatial = 1
            self.flat_size = in_channels * spatial * spatial

            self.classifier = nn.Sequential(
                nn.Linear(self.flat_size, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 10),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    model = SmallCNN().to(device)

    # --- Optimizer + scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    estimated_epochs = max(1, RUN_TIMEOUT_MINUTES)
    total_steps = len(train_loader) * estimated_epochs
    if lr_schedule == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, total_steps
        )
    else:
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=total_steps, gamma=1.0
        )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_steps
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_steps]
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # --- Training loop ---
    best_val_loss = float("inf")
    best_time_seconds = 0.0
    best_progress_percent = 0.0
    start_time = time.monotonic()
    max_time_seconds = RUN_TIMEOUT_MINUTES * 60
    epoch = 0

    while True:
        model.train()
        train_losses = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            if gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_losses.append(loss.item())
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = np.mean(val_losses)
        accuracy = correct / total if total > 0 else 0

        elapsed_time_seconds = time.monotonic() - start_time
        progress_percent = min((elapsed_time_seconds / max_time_seconds) * 100, 100.0)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_time_seconds = elapsed_time_seconds
            best_progress_percent = progress_percent

        wandb.log(
            {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "accuracy": accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_time_seconds": elapsed_time_seconds,
                "progress_percent": progress_percent,
            }
        )

        print(
            f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}, acc={accuracy:.4f}, "
            f"time={elapsed_time_seconds:.1f}s, progress={progress_percent:.1f}%"
        )

        epoch += 1
        if elapsed_time_seconds >= max_time_seconds:
            break

    wandb.finish()

    return {
        "run_id": run_id,
        "best_val_loss": float(best_val_loss),
        "best_time_seconds": float(best_time_seconds),
        "best_progress_percent": float(best_progress_percent),
        "final_val_loss": float(avg_val_loss),
        "final_accuracy": float(accuracy),
    }
