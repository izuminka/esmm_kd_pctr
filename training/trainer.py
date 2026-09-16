"""Generic Trainer for Ali-CCP recommendation models (DCN, ESMM, ...).

Trains on a single device (GPU if available, else CPU).

Usage (via scripts/train.py — don't instantiate directly in tests):
    trainer = Trainer(cfg, root_dir)
    trainer.fit(model, dataset, metric_fn)
"""

import logging
import os
import random
import time
import warnings
from dataclasses import dataclass

import pandas as pd
import torch

from data.vocab import encode_batch

logger = logging.getLogger(__name__)


def resolve_device(gpus: int) -> torch.device:
    """Pick the single device to train on.

    gpus == 0 forces CPU. Any other value uses cuda:0 if available,
    falling back to CPU with a warning otherwise.
    """
    if gpus == 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        warnings.warn(
            "CUDA requested but not available. Falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    return torch.device("cuda:0")


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
    return {k: tuple(t.to(device, non_blocking=True) for t in v) for k, v in encoded.items()}


# y/z are 0/1 labels; pin them so the schema does not depend on whether the
# caller came from the tensor path (floats) or the Impression path (ints).
_LABEL_DTYPES = {"y": "int8", "z": "int8"}


def save_predictions(ids, ys, zs, scores, path: str) -> None:
    pd.DataFrame(
        {"sample_id": ids, "y": ys, "z": zs, "score": scores}
    ).astype({**_LABEL_DTYPES, "score": "float32"}).to_parquet(path, index=False)


def save_esmm_predictions(ids, ys, zs, q_scores, v_scores, path: str) -> None:
    pd.DataFrame(
        {"sample_id": ids, "y": ys, "z": zs, "q": q_scores, "v": v_scores}
    ).astype({**_LABEL_DTYPES, "q": "float32", "v": "float32"}).to_parquet(path, index=False)


class Trainer:
    def __init__(self, cfg: dict, root_dir: str):
        tr = cfg["training"]
        self.model_name = cfg["model"]
        self.epochs = tr["epochs"]
        self.batch_size = tr["batch_size"]
        self.eval_batch_size = tr["eval_batch_size"]
        self.lr = tr["lr"]
        self.seed = tr.get("seed", 42)
        self.val_frac = tr["val_frac"]
        self.out_dir = os.path.join(root_dir, tr["out_dir"])
        self.device = resolve_device(tr.get("gpus", -1))
        es = tr.get("early_stopping", {})
        self.es_patience = es.get("patience", 5)
        self.es_min_delta = es.get("min_delta", 1e-4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, model, dataset, metric_fn) -> str:
        """Train model, checkpoint best by metric_fn, export predictions.

        Accepts either a NumpyDataset (preferred, no Python Impression objects)
        or the legacy Dataset (list-of-Impression).

        metric_fn(model, encoded_val, val_y) -> float  (higher = better).
        Returns the path of the saved checkpoint.
        """
        from training.dataset import NumpyDataset
        use_numpy = isinstance(dataset, NumpyDataset)

        random.seed(self.seed)
        os.makedirs(self.out_dir, exist_ok=True)

        device = self.device
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        checkpoint_path = os.path.join(self.out_dir, f"{self.model_name}_model.pt")

        if not use_numpy:
            val_encoded = _to_device(
                encode_batch(dataset.val, dataset.field_ids, dataset.vocabs), device
            )
            val_y = torch.tensor(
                [imp.y for imp in dataset.val], dtype=torch.float32, device=device
            )

        vocabs = dataset.vocab if use_numpy else dataset.vocabs
        field_ids = dataset.field_ids

        # The encoded split is authoritative: evaluate.py/predict.py re-slice raw
        # impressions by the checkpoint's val_frac, so recording the config's
        # value instead would send them to rows the model trained on.
        val_frac = dataset.val_frac if use_numpy else self.val_frac
        if use_numpy and val_frac != self.val_frac:
            logger.warning(
                "config val_frac=%s but the encoded data was split at %s; using %s. "
                "Re-run preprocess.py to change the split.",
                self.val_frac, val_frac, val_frac,
            )

        logger.info(
            "Training %s | %d epochs | batch=%d | lr=%s | device=%s | early_stopping(patience=%d, min_delta=%s)",
            self.model_name, self.epochs, self.batch_size, self.lr, self.device,
            self.es_patience, self.es_min_delta,
        )

        best_score = -float("inf")
        no_improve = 0
        for epoch in range(1, self.epochs + 1):
            t0 = time.perf_counter()
            if use_numpy:
                loss = self._train_epoch_numpy(model, dataset.train, optimizer)
            else:
                loss = self.train_epoch(model, dataset.train, dataset.vocabs, field_ids, optimizer)
            epoch_secs = time.perf_counter() - t0

            if use_numpy:
                score = self._val_score_numpy(model, dataset.val)
            else:
                score = metric_fn(model, val_encoded, val_y)

            logger.info(
                "Epoch %d/%d  loss=%.4f  val_score=%.4f  time=%.1fs",
                epoch, self.epochs, loss, score, epoch_secs,
            )
            if score > best_score + self.es_min_delta:
                best_score = score
                no_improve = 0
                ckpt_data = {
                    "model_type": self.model_name,
                    "model_state_dict": model.state_dict(),
                    "vocabs": vocabs,
                    "field_ids": field_ids,
                    "vocab_sizes": dataset.vocab_sizes,
                    "val_frac": val_frac,
                }
                if self.model_name == "esmm":
                    ckpt_data["model_params"] = {
                        "emb_dim": model.emb_dim,
                        "tower_dims": model.tower_dims,
                    }
                torch.save(ckpt_data, checkpoint_path)
                logger.info("  -> checkpoint saved (val_score=%.4f)", score)
            else:
                no_improve += 1
                logger.info("  no improvement (%d/%d)", no_improve, self.es_patience)
                if no_improve >= self.es_patience:
                    logger.info("Early stopping triggered after %d epochs with no improvement.", self.es_patience)
                    break

        logger.info("Best val score: %.4f. Loading best checkpoint for export...", best_score)
        ckpt = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        logger.info("Exporting predictions...")
        if use_numpy:
            splits = [("train", dataset.train), ("val", dataset.val), ("test", dataset.test)]
            for split_name, split_data in splits:
                if split_data is None:  # split not loaded (see NumpyDataset.load(load_test=))
                    logger.info("  %s: not loaded, skipping export", split_name)
                    continue
                t0 = time.perf_counter()
                path = os.path.join(self.out_dir, f"predictions_{split_name}.parquet")
                ids, ys, zs, scores = self._predict_numpy(model, split_data)
                save_predictions(ids, ys, zs, scores, path)
                logger.info("  %s: %d rows -> %s (%.1fs)", split_name, len(ids), path, time.perf_counter() - t0)
        else:
            for split_name, imps in [("train", dataset.train), ("val", dataset.val), ("test", dataset.test)]:
                t0 = time.perf_counter()
                path = os.path.join(self.out_dir, f"predictions_{split_name}.parquet")
                if self.model_name == "esmm":
                    ids, ys, zs, qs, vs = self.predict_esmm(model, imps, dataset.vocabs, field_ids)
                    save_esmm_predictions(ids, ys, zs, qs, vs, path)
                else:
                    ids, ys, zs, scores = self.predict(model, imps, dataset.vocabs, field_ids)
                    save_predictions(ids, ys, zs, scores, path)
                logger.info("  %s: %d rows -> %s (%.1fs)", split_name, len(ids), path, time.perf_counter() - t0)

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
        device = self.device
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

    def predict_esmm(self, model, impressions: list, vocabs: dict, field_ids: list) -> tuple:
        """Return (sample_ids, ys, zs, q_scores, v_scores) as plain Python lists."""
        model.eval()
        device = self.device
        ids, ys, zs, q_scores, v_scores = [], [], [], [], []
        with torch.no_grad():
            for start in range(0, len(impressions), self.batch_size):
                batch = impressions[start : start + self.batch_size]
                enc = _to_device(encode_batch(batch, field_ids, vocabs), device)
                q, v = model(enc)
                ids.extend(imp.sample_id for imp in batch)
                ys.extend(imp.y for imp in batch)
                zs.extend(imp.z for imp in batch)
                q_scores.extend(q.cpu().tolist())
                v_scores.extend(v.cpu().tolist())
        return ids, ys, zs, q_scores, v_scores

    # ------------------------------------------------------------------
    # NumpyDataset-backed epoch/predict/val helpers
    # ------------------------------------------------------------------

    def _train_epoch_numpy(self, model, split_data, optimizer) -> float:
        """One shuffled epoch over a SplitData; returns mean batch loss."""
        model.train()
        n = len(split_data)
        indices = list(range(n))
        random.shuffle(indices)
        total_loss, n_batches = 0.0, 0

        for start in range(0, n, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            optimizer.zero_grad()
            loss_val = self._train_step_numpy(model, split_data, batch_idx)
            optimizer.step()
            total_loss += loss_val
            n_batches += 1

            if n_batches % 500 == 0:
                logger.debug("  batch %d/%d  loss=%.4f", n_batches, n // self.batch_size, loss_val)

        return total_loss / n_batches if n_batches else 0.0

    def _train_step_numpy(self, model, split_data, batch_idx: list) -> float:
        """Forward + backward for a NumpyDataset batch.

        No host->device transfer: the split already lives on self.device.
        """
        from models.dcn import dcn_loss

        enc, y, _ = split_data.get_batch(batch_idx)
        loss = dcn_loss(y, model(enc))
        loss.backward()
        return loss.item()

    def _predict_numpy(self, model, split_data) -> tuple:
        """Return (sample_ids, ys, zs, scores) as numpy arrays from a SplitData."""
        model.eval()
        ids_out, ys_out, zs_out, scores_out = [], [], [], []
        n = len(split_data)
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                idx = torch.arange(start, min(start + self.batch_size, n),
                                   device=split_data.device)
                enc, y, z = split_data.get_batch(idx)
                logit = model(enc)
                ids_out.append(split_data.sample_ids[idx].cpu())
                ys_out.append(y.cpu())
                zs_out.append(z.cpu())
                scores_out.append(torch.sigmoid(logit).cpu())
        return (
            torch.cat(ids_out).numpy(),
            torch.cat(ys_out).numpy(),
            torch.cat(zs_out).numpy(),
            torch.cat(scores_out).numpy(),
        )

    def _val_score_numpy(self, model, val_split) -> float:
        """Compute validation AUC in batches (bounds activation memory)."""
        from evaluation.metrics import compute_metrics
        model.eval()
        all_scores, all_y = [], []
        eval_batch = self.eval_batch_size
        n = len(val_split)
        with torch.no_grad():
            for start in range(0, n, eval_batch):
                idx = torch.arange(start, min(start + eval_batch, n),
                                   device=val_split.device)
                enc, y, _ = val_split.get_batch(idx)
                logit = model(enc)
                all_scores.append(torch.sigmoid(logit).cpu())
                all_y.append(y.cpu())
        scores = torch.cat(all_scores).numpy()
        ys = torch.cat(all_y).numpy()
        return compute_metrics(scores, ys)["auc"]

    # ------------------------------------------------------------------
    # Internal: forward + backward
    # ------------------------------------------------------------------

    def _train_step(self, model, batch: list, vocabs: dict, field_ids: list) -> float:
        """Forward + backward.

        Gradient zeroing and optimizer.step() are the caller's responsibility.
        Returns the scalar loss value for logging.
        """
        if self.model_name == "esmm":
            return self._train_step_esmm(model, batch, vocabs, field_ids)

        from models.dcn import dcn_loss  # late import keeps trainer model-agnostic

        device = self.device
        enc = _to_device(encode_batch(batch, field_ids, vocabs), device)
        y = torch.tensor([imp.y for imp in batch], dtype=torch.float32, device=device)
        loss = dcn_loss(y, model(enc))
        loss.backward()
        return loss.item()

    def _train_step_esmm(self, model, batch: list, vocabs: dict, field_ids: list) -> float:
        from models.esmm import esmm_loss

        device = self.device
        enc = _to_device(encode_batch(batch, field_ids, vocabs), device)
        y = torch.tensor([imp.y for imp in batch], dtype=torch.float32, device=device)
        z = torch.tensor([imp.z for imp in batch], dtype=torch.float32, device=device)
        q, v = model(enc)
        loss = esmm_loss(y, z, q, v)
        loss.backward()
        return loss.item()
