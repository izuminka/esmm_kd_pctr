"""Generic Trainer for Ali-CCP recommendation models (DCN, ESMM, ...).

Multi-GPU support: when multiple CUDA devices are available, each mini-batch is
split into equal per-device chunks.  Forwards run concurrently (CUDA kernels
are queued asynchronously while the CPU encodes the next chunk), and per-replica
gradients are reduced back to the master model before each optimizer step.
Falls back to single GPU or CPU automatically.

Usage (via scripts/train.py — don't instantiate directly in tests):
    trainer = Trainer(cfg, root_dir)
    trainer.fit(model, dataset, metric_fn)
"""

import os
import random
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn.parallel as parallel

from data.vocab import encode_batch


@dataclass
class Dataset:
    train: list
    val: list
    test: list
    vocabs: dict
    field_ids: list

    @property
    def vocab_sizes(self) -> dict:
        return {f: len(v) for f, v in self.vocabs.items()}


def _to_device(encoded: dict, device: torch.device) -> dict:
    return {k: tuple(t.to(device) for t in v) for k, v in encoded.items()}


def save_predictions(ids, ys, zs, scores, path: str) -> None:
    pd.DataFrame(
        {"sample_id": ids, "y": ys, "z": zs, "score": scores}
    ).astype({"score": "float32"}).to_parquet(path, index=False)


class Trainer:
    def __init__(self, cfg: dict, root_dir: str):
        tr = cfg["training"]
        self.model_name = cfg["model"]
        self.epochs = tr["epochs"]
        self.batch_size = tr["batch_size"]
        self.lr = tr["lr"]
        self.seed = tr.get("seed", 42)
        self.val_frac = tr["val_frac"]
        self.out_dir = os.path.join(root_dir, tr["out_dir"])
        self.devices = self._setup_devices(tr.get("gpus", -1))
        es = tr.get("early_stopping", {})
        self.es_patience = es.get("patience", 5)
        self.es_min_delta = es.get("min_delta", 1e-4)

    def _setup_devices(self, gpus: int) -> list:
        if gpus == 0:
            return [torch.device("cpu")]
        if not torch.cuda.is_available():
            if gpus != 0:
                import warnings
                warnings.warn(
                    "CUDA requested but not available. Falling back to CPU.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return [torch.device("cpu")]
        n_available = torch.cuda.device_count()
        n = n_available if gpus == -1 else min(gpus, n_available)
        return [torch.device(f"cuda:{i}") for i in range(max(1, n))]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, model, dataset: Dataset, metric_fn) -> str:
        """Train model, checkpoint best by metric_fn, export predictions.

        metric_fn(model, encoded_val, val_y) -> float  (higher = better).
        Returns the path of the saved checkpoint.
        """
        random.seed(self.seed)
        os.makedirs(self.out_dir, exist_ok=True)

        device = self.devices[0]
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        checkpoint_path = os.path.join(self.out_dir, f"{self.model_name}_model.pt")

        val_encoded = _to_device(encode_batch(dataset.val, dataset.field_ids, dataset.vocabs), device)
        val_y = torch.tensor([imp.y for imp in dataset.val], dtype=torch.float32, device=device)

        best_score = -float("inf")
        no_improve = 0
        for epoch in range(1, self.epochs + 1):
            loss = self.train_epoch(model, dataset.train, dataset.vocabs, dataset.field_ids, optimizer)
            score = metric_fn(model, val_encoded, val_y)
            print(f"Epoch {epoch}/{self.epochs}  loss={loss:.4f}  val_score={score:.4f}")
            if score > best_score + self.es_min_delta:
                best_score = score
                no_improve = 0
                torch.save(
                    {
                        "model_type": self.model_name,
                        "model_state_dict": model.state_dict(),
                        "vocabs": dataset.vocabs,
                        "field_ids": dataset.field_ids,
                        "vocab_sizes": dataset.vocab_sizes,
                        "val_frac": self.val_frac,
                    },
                    checkpoint_path,
                )
                print(f"  -> checkpoint saved (val_score={score:.4f})")
            else:
                no_improve += 1
                if no_improve >= self.es_patience:
                    print(f"Early stopping: no improvement for {self.es_patience} epochs.")
                    break

        print(f"\nBest val score: {best_score:.4f}. Loading best checkpoint for export...")
        ckpt = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        print("Exporting predictions...")
        for split_name, imps in [("train", dataset.train), ("val", dataset.val), ("test", dataset.test)]:
            ids, ys, zs, scores = self.predict(model, imps, dataset.vocabs, dataset.field_ids)
            path = os.path.join(self.out_dir, f"predictions_{split_name}.parquet")
            save_predictions(ids, ys, zs, scores, path)
            print(f"  {split_name}: {len(ids)} rows -> {path}")

        return checkpoint_path

    def train_epoch(self, model, impressions: list, vocabs: dict, field_ids: list, optimizer) -> float:
        """One shuffled training epoch; returns mean batch loss."""
        model.train()
        indices = list(range(len(impressions)))
        random.shuffle(indices)
        total_loss, n_batches = 0.0, 0

        for start in range(0, len(indices), self.batch_size):
            batch = [impressions[i] for i in indices[start : start + self.batch_size]]
            optimizer.zero_grad()
            loss_val = self._train_step(model, batch, vocabs, field_ids)
            optimizer.step()
            total_loss += loss_val
            n_batches += 1

        return total_loss / n_batches if n_batches else 0.0

    def predict(self, model, impressions: list, vocabs: dict, field_ids: list) -> tuple:
        """Return (sample_ids, ys, zs, scores) as plain Python lists."""
        model.eval()
        device = self.devices[0]
        ids, ys, zs, scores = [], [], [], []
        with torch.no_grad():
            for start in range(0, len(impressions), self.batch_size):
                batch = impressions[start : start + self.batch_size]
                enc = _to_device(encode_batch(batch, field_ids, vocabs), device)
                logit = model(enc)
                ids.extend(imp.sample_id for imp in batch)
                ys.extend(imp.y for imp in batch)
                zs.extend(imp.z for imp in batch)
                scores.extend(torch.sigmoid(logit).cpu().tolist())
        return ids, ys, zs, scores

    # ------------------------------------------------------------------
    # Internal: forward + backward (+ grad reduction for multi-GPU)
    # ------------------------------------------------------------------

    def _train_step(self, model, batch: list, vocabs: dict, field_ids: list) -> float:
        """Forward + backward (and grad reduction for multi-GPU).

        Gradient zeroing and optimizer.step() are the caller's responsibility.
        Returns the scalar loss value for logging.
        """
        from models.dcn import dcn_loss  # late import keeps trainer model-agnostic

        if len(self.devices) == 1:
            device = self.devices[0]
            enc = _to_device(encode_batch(batch, field_ids, vocabs), device)
            y = torch.tensor([imp.y for imp in batch], dtype=torch.float32, device=device)
            loss = dcn_loss(y, model(enc))
            loss.backward()
            return loss.item()

        # Multi-GPU path: split batch into per-device chunks.
        # CUDA forwards queue asynchronously, so the CPU can encode the next
        # chunk while the GPU is still processing the previous one.
        n_gpus = len(self.devices)
        chunk_size = max(1, (len(batch) + n_gpus - 1) // n_gpus)
        chunks = [batch[i : i + chunk_size] for i in range(0, len(batch), chunk_size)]
        active_devices = self.devices[: len(chunks)]
        replicas = parallel.replicate(model, active_devices)

        logits, ys = [], []
        for chunk, replica, device in zip(chunks, replicas, active_devices):
            enc = _to_device(encode_batch(chunk, field_ids, vocabs), device)
            logits.append(replica(enc))
            ys.append(torch.tensor([imp.y for imp in chunk], dtype=torch.float32, device=device))

        # Gather to primary device and compute a single scalar loss.
        logit_all = torch.cat([l.to(self.devices[0]) for l in logits])
        y_all = torch.cat([y.to(self.devices[0]) for y in ys])
        loss = dcn_loss(y_all, logit_all)

        # Backward propagates through gather back to each replica's params.
        loss.backward()

        # Reduce per-replica gradients to the master model parameters so
        # optimizer.step() (which acts on master params) sees the full gradient.
        for master_p, *replica_ps in zip(model.parameters(), *[r.parameters() for r in replicas]):
            grads = [p.grad.to(self.devices[0]) for p in replica_ps if p.grad is not None]
            master_p.grad = (sum(grads) / len(grads)) if grads else None

        return loss.item()
