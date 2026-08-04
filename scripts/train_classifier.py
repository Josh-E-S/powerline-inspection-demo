"""Fine-tune EfficientNetV2-S on the InsPLAD fault crops.

One softmax head covers all 11 asset__condition classes; at inference the
app masks the logits down to the conditions valid for whichever asset was
detected.

The fault splits are badly imbalanced, with one class holding 5,742 test
crops against another's 20. Batches are therefore class-balanced during
training, and model selection uses balanced accuracy rather than plain
accuracy, which would be dominated by the largest class.

Resumable in the same way as the detector script.

    python scripts/train_classifier.py
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "fault"
SEED = 42


def device():
    """Pick the best available accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def loaders(batch_size, workers, smoke=False):
    """Build the train and validation loaders.

    Training samples are drawn with weights inverse to class frequency,
    so each batch sees a roughly even spread of conditions.
    """
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_ds = datasets.ImageFolder(DATA / "train", train_tf)
    val_ds = datasets.ImageFolder(DATA / "val", eval_tf)

    counts = torch.bincount(torch.tensor(train_ds.targets))
    sample_weights = (1.0 / counts.float())[torch.tensor(train_ds.targets)]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=1024 if smoke else len(train_ds),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    return train_dl, val_dl, train_ds.classes


@torch.no_grad()
def balanced_accuracy(model, dl, n_classes, dev):
    """Mean per-class recall, so every class counts equally."""
    model.eval()
    correct = torch.zeros(n_classes)
    total = torch.zeros(n_classes)
    for x, y in dl:
        pred = model(x.to(dev)).argmax(1).cpu()
        for cls in range(n_classes):
            mask = y == cls
            total[cls] += mask.sum()
            correct[cls] += (pred[mask] == cls).sum()
    recalls = correct[total > 0] / total[total > 0]
    return recalls.mean().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--patience",
        type=int,
        default=10,
        help="early stop after N epochs without val improvement",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="cls_effv2s")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument(
        "--runs-dir",
        default=str(REPO / "runs"),
        help="checkpoint/metrics root; point at a persistent "
        "mount (e.g. Google Drive) on ephemeral instances",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="pipeline check: 2 epochs on ~1k sampled crops",
    )
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 2
        if args.name == "cls_effv2s":
            args.name = "cls_effv2s_smoke"

    torch.manual_seed(SEED)
    random.seed(SEED)
    dev = device()
    run_dir = Path(args.runs_dir) / "classify" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"

    train_dl, val_dl, classes = loaders(args.batch, args.workers, args.smoke)
    n = len(classes)

    model = models.efficientnet_v2_s(
        weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n)
    model.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    start_epoch, best_bacc, since_best = 0, 0.0, 0
    if last_path.exists() and not args.fresh:
        ckpt = torch.load(last_path, map_location=dev, weights_only=True)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_epoch, best_bacc = ckpt["epoch"] + 1, ckpt["best_bacc"]
        since_best = ckpt.get("since_best", 0)
        print(f"Resuming at epoch {start_epoch} (best {best_bacc:.4f})")

    log = open(run_dir / "metrics.jsonl", "a")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
            seen += len(y)
        sched.step()

        bacc = balanced_accuracy(model, val_dl, n, dev)
        row = {
            "epoch": epoch,
            "train_loss": running / seen,
            "val_balanced_acc": round(bacc, 4),
        }
        print(row)
        log.write(json.dumps(row) + "\n")
        log.flush()

        if bacc > best_bacc:
            best_bacc, since_best = bacc, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": classes,
                    "arch": "efficientnet_v2_s",
                },
                best_path,
            )
        else:
            since_best += 1
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "epoch": epoch,
                "best_bacc": best_bacc,
                "since_best": since_best,
                "classes": classes,
                "arch": "efficientnet_v2_s",
            },
            last_path,
        )
        if since_best >= args.patience:
            print(f"Early stop: no val improvement in {args.patience} epochs.")
            break

    print(f"Done. Best val balanced accuracy: {best_bacc:.4f} -> {best_path}")


if __name__ == "__main__":
    main()
