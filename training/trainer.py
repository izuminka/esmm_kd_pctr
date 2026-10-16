"""Generic Trainer for Ali-CCP recommendation models (DCN, ESMM, ...).

Trains on a single device (GPU if available, else CPU).

Usage (via scripts/train.py — don't instantiate directly in tests):
    trainer = Trainer(cfg, root_dir)
    trainer.fit(model, dataset, metric_fn)
"""

import csv
import json
import logging
import os
import random
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from data.vocab import encode_batch
from training import downsampling
from training.downsampling import NONCLICK_SAMPLING_KEY

logger = logging.getLogger(__name__)

# Recorded in every checkpoint so a model states how its data was split.
SPLIT_RULE = "validation iff splitmix64(sample_id xor 42) < 1844674407370955264"


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


# Use the same population names and metrics as the final split reports.
_HISTORY_POPULATIONS = {"CTR": "ctr", "CTCVR": "ctcvr", "CTCVR-logged": "ctcvr_logged", "pCVR": "pcvr"}
_HISTORY_METRICS = {"auc": "auc", "pr_auc": "prauc", "ne": "ne", "ce": "ce"}


# y/z are 0/1 labels; pin them so the schema does not depend on whether the
# caller came from the tensor path (floats) or the Impression path (ints).
_LABEL_DTYPES = {"y": "int8", "z": "int8"}


def save_scores(ids, ys, zs, scores: dict, path: str) -> None:
    """Write one split's predictions; `scores` maps column name -> values.

    A model contributes as many score columns as it has reported heads, so the
    column set is the model's, not this writer's: downstream evaluation reads
    them back by name.
    """
    pd.DataFrame({"sample_id": ids, "y": ys, "z": zs, **scores}).astype(
        {**_LABEL_DTYPES, **{c: "float32" for c in scores}}
    ).to_parquet(path, index=False)


def save_predictions(ids, ys, zs, scores, path: str) -> None:
    save_scores(ids, ys, zs, {"score": scores}, path)


def save_esmm_predictions(ids, ys, zs, q_scores, v_scores, path: str) -> None:
    save_scores(ids, ys, zs, {"q": q_scores, "v": v_scores}, path)


def write_split(run_dir: str, split: str, model: str, ids, ys, zs, scores: dict,
                save_predictions: bool) -> str:
    """Write one split's predictions, or only its metrics; returns the path written.

    With predictions, writes <run>/predictions/<split>.parquet. Without, scores the
    split in memory and writes its metric rows to <run>/metrics/<split>.json, which
    scripts/report_metrics.py reads in place of the parquet. Labels and scores are
    cast to the parquet's dtypes first, so both routes compute identical metrics.
    """
    if save_predictions:
        path = os.path.join(run_dir, "predictions", f"{split}.parquet")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_scores(ids, ys, zs, scores, path)
        return path

    from scripts.report_metrics import metric_rows

    path = os.path.join(run_dir, "metrics", f"{split}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = metric_rows(
        split, model,
        np.asarray(ys).astype(_LABEL_DTYPES["y"]),
        np.asarray(zs).astype(_LABEL_DTYPES["z"]),
        {c: np.asarray(s, dtype=np.float32) for c, s in scores.items()},
    )
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    return path


class Trainer:
    def __init__(self, cfg: dict, root_dir: str):
        from training.downsampling import DEFAULT_NONCLICK_RATE, from_config
        self.nonclick_sampling = from_config(cfg)
        self.nonclick_rate = self.nonclick_sampling["rate"] if self.nonclick_sampling else DEFAULT_NONCLICK_RATE
        tr = cfg["training"]
        self.model_name = cfg["model"]
        self.epochs = tr["epochs"]
        self.batch_size = tr["batch_size"]
        self.eval_batch_size = tr["eval_batch_size"]
        self.lr = tr["lr"]
        self.embedding_lr = tr.get("embedding_lr")
        if self.embedding_lr is not None:
            self.embedding_lr = float(self.embedding_lr)
            if self.model_name != "dcn":
                raise ValueError("embedding_lr is currently supported only for DCN")
            if not np.isfinite(self.embedding_lr) or self.embedding_lr <= 0:
                raise ValueError("embedding_lr must be finite and positive")
        self.weight_decay = tr.get("weight_decay", 0.0)
        self.seed = tr.get("seed", 42)
        self.grad_clip_norm = tr["grad_clip_norm"]
        self.shuffle_block_size = tr["shuffle_block_size"]
        # Which validation metric ranks checkpoints. It is a property of the arm,
        # not of the architecture: the recipe selects the click-only model on its
        # CTR head and ESMM on its native product.
        self.checkpoint_metric = tr.get("checkpoint_metric", "ctr_prauc")
        # "last" is a fixed-epoch experiment: train the full budget and export
        # its final weights. "best" may stop early and restores the best score.
        self.keep_checkpoint = tr.get("keep_checkpoint", "best")
        if self.keep_checkpoint not in ("best", "last"):
            raise ValueError(f"keep_checkpoint must be 'best' or 'last', got {self.keep_checkpoint!r}")
        from scripts.report_metrics import (
            ESMM_MODELS, LOGGED_PCTR_TRAINING_MODELS, logged_ctcvr_splits,
        )
        self.is_esmm = self.model_name in ESMM_MODELS
        self.uses_logged_pctr_for_training = self.model_name in LOGGED_PCTR_TRAINING_MODELS
        self.is_cnn_attention = self.model_name == "cnn_attention"
        self.cold_start_validation = tr.get("cold_start_validation")
        if self.cold_start_validation is not None:
            if not self.is_cnn_attention:
                raise ValueError("cold_start_validation is supported only for cnn_attention")
            self.cold_start_validation = {
                "field": str(self.cold_start_validation["field"]),
                "missing_rate": float(self.cold_start_validation["missing_rate"]),
                "seed": int(self.cold_start_validation.get("seed", self.seed)),
            }
            if not 0 <= self.cold_start_validation["missing_rate"] <= 1:
                raise ValueError("cold-start missing_rate must lie in [0, 1]")
        cnn_metrics = (
            "ctr_auc", "ctr_ne", "ctr_prauc",
            "cold_ctr_auc", "cold_ctr_ne", "cold_ctr_prauc",
        )
        if self.is_cnn_attention and self.checkpoint_metric not in cnn_metrics:
            raise ValueError(
                "cnn_attention checkpoint_metric must be a CTR or cold-start CTR metric")
        if self.checkpoint_metric.startswith("cold_") and self.cold_start_validation is None:
            raise ValueError(
                f"checkpoint_metric {self.checkpoint_metric!r} requires cold_start_validation")
        # Logged-score models validate on p*v; native ESMM only reports it on test.
        self.logged_ctcvr_splits = logged_ctcvr_splits(cfg)
        self.is_logged_ctcvr = bool(self.logged_ctcvr_splits)
        # Where the logged pCTR came from (see data.logged_pctr); the caller sets it
        # when a config names one, and it is saved with every checkpoint.
        self.logged_pctr_source = None
        # False scores each split in memory instead of exporting its predictions.
        self.save_predictions = tr.get("save_predictions", True)
        self.out_dir = os.path.join(root_dir, tr["out_dir"])
        self.device = resolve_device(tr.get("gpus", -1))
        es = tr.get("early_stopping", {})
        self.es_enabled = self.keep_checkpoint == "best" and es.get("enabled", True)
        self.es_patience = es.get("patience", 5)
        self.es_min_delta = es.get("min_delta", 1e-4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _make_optimizer(self, model):
        if self.embedding_lr is None:
            parameters = model.parameters()
        else:
            embedding_parameters = list(model.embeddings.parameters())
            embedding_ids = {id(p) for p in embedding_parameters}
            dense_parameters = [p for p in model.parameters() if id(p) not in embedding_ids]
            parameters = [
                {"params": embedding_parameters, "lr": self.embedding_lr},
                {"params": dense_parameters, "lr": self.lr},
            ]
            logger.info("Optimizer rates: embeddings=%s dense=%s weight_decay=%s",
                        self.embedding_lr, self.lr, self.weight_decay)
        return torch.optim.Adam(parameters, lr=self.lr, weight_decay=self.weight_decay)

    def fit(self, model, dataset, metric_fn) -> str:
        """Train model, checkpoint best by metric_fn, export predictions.

        Accepts either a NumpyDataset (preferred, no Python Impression objects)
        or the legacy Dataset (list-of-Impression).

        Legacy datasets use metric_fn(model, encoded_val, val_y), with val_z
        as a fourth argument for ESMM. NE metrics are minimized; other
        checkpoint metrics are maximized.
        Returns the path of the saved checkpoint.
        """
        from training.dataset import NumpyDataset
        use_numpy = isinstance(dataset, NumpyDataset)
        if self.nonclick_sampling is not None and not use_numpy:
            raise ValueError("nonclick sampling training requires the encoded dataset pipeline")
        if self.is_cnn_attention and not use_numpy:
            raise ValueError("cnn_attention requires the encoded NumpyDataset pipeline")

        needed = set(self.logged_ctcvr_splits)
        if self.uses_logged_pctr_for_training:
            needed.add("train")
        for split_name in needed:
            split = getattr(dataset, split_name)
            if split is not None and not (use_numpy and split.logged_pctr is not None):
                raise ValueError(f"{self.model_name} needs logged pCTR on the encoded {split_name} split")
        if self.checkpoint_metric in ("logged_ctcvr_prauc", "logged_ctcvr_ne") and "val" not in self.logged_ctcvr_splits:
            raise ValueError("validation p*v requires a logged-pCTR training model; native ESMM validates on q*v")

        random.seed(self.seed)
        os.makedirs(self.out_dir, exist_ok=True)

        device = self.device
        model = model.to(device)
        if (
            self.cold_start_validation is not None
            and self.cold_start_validation["field"] not in model.field_ids
        ):
            raise ValueError(
                "cold_start_validation names unknown field "
                f"{self.cold_start_validation['field']!r}")
        optimizer = self._make_optimizer(model)
        checkpoint_path = os.path.join(self.out_dir, "best.pt")
        history_path = os.path.join(self.out_dir, "history.csv")
        history_extras = list(
            f"val_{population}_{metric}"
            for name, population in _HISTORY_POPULATIONS.items()
            if name != "CTCVR-logged" or "val" in self.logged_ctcvr_splits
            if not self.is_cnn_attention or name in ("CTR", "CTCVR")
            for metric in _HISTORY_METRICS.values()
        ) if self.is_esmm or self.is_cnn_attention else []
        if self.is_cnn_attention:
            history_extras.extend((
                "val_attention_click_mean",
                "val_attention_conversion_mean",
                "val_attention_entropy",
                "val_attention_min_task_weight",
            ))
            if self.cold_start_validation is not None:
                history_extras.extend(
                    f"val_cold_ctr_{metric}" for metric in _HISTORY_METRICS.values())
        history_extras = tuple(history_extras)
        with open(history_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss", "val_score", "seconds", "saved", *history_extras])

        if not use_numpy:
            val_encoded = _to_device(
                encode_batch(dataset.val, dataset.field_ids, dataset.vocabs), device
            )
            val_y = torch.tensor(
                [imp.y for imp in dataset.val], dtype=torch.float32, device=device
            )

        vocabs = dataset.vocab if use_numpy else dataset.vocabs
        field_ids = dataset.field_ids


        logger.info(
            "Training %s | %d epochs | batch=%d | lr=%s | device=%s | checkpoint=%s | early_stopping(enabled=%s, patience=%d, min_delta=%s)",
            self.model_name, self.epochs, self.batch_size, self.lr, self.device,
            self.keep_checkpoint, self.es_enabled, self.es_patience, self.es_min_delta,
        )

        minimize = self.checkpoint_metric.endswith("_ne")
        best_score = float("inf") if minimize else -float("inf")
        no_improve = 0
        for epoch in range(1, self.epochs + 1):
            t0 = time.perf_counter()
            if use_numpy:
                loss = self._train_epoch_numpy(model, dataset.train, optimizer, epoch - 1)
            else:
                loss = self.train_epoch(model, dataset.train, dataset.vocabs, field_ids, optimizer)
            epoch_secs = time.perf_counter() - t0

            if use_numpy:
                score, extras = self._val_score_numpy(model, dataset.val)
            else:
                if self.is_esmm:
                    val_z = torch.tensor([imp.z for imp in dataset.val], dtype=torch.float32, device=device)
                    score, extras = metric_fn(model, val_encoded, val_y, val_z), {}
                else:
                    score, extras = metric_fn(model, val_encoded, val_y), {}

            logger.info(
                "Epoch %d/%d  loss=%.4f  val_score=%.4f  time=%.1fs %s",
                epoch, self.epochs, loss, score, epoch_secs,
                "  ".join(f"{k}={v:.6f}" for k, v in extras.items()),
            )
            improved = (score < best_score - self.es_min_delta if minimize
                        else score > best_score + self.es_min_delta)
            # Last mode saves every epoch and always runs the full epoch budget.
            saved = improved or self.keep_checkpoint == "last"
            # Appended per epoch so a run that dies mid-training keeps its history.
            with open(history_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, loss, score, round(epoch_secs, 1), saved,
                                        *(extras.get(c, "") for c in history_extras)])
            if saved:
                ckpt_data = {
                    "model_type": self.model_name,
                    "model_state_dict": model.state_dict(),
                    "vocabs": vocabs,
                    "field_ids": field_ids,
                    "vocab_sizes": dataset.vocab_sizes,
                    "split_rule": SPLIT_RULE,
                }
                if self.is_esmm:
                    ckpt_data["model_params"] = {
                        "emb_dim": model.emb_dim,
                        "tower_dims": model.tower_dims,
                    }
                    if self.model_name in ("esmm", "esmm_logged_pctr"):
                        ckpt_data["model_params"]["aux_weight"] = model.aux_weight
                if self.model_name == "esmm_logged_pctr":
                    ckpt_data["model_params"].update(
                        objective=model.objective, alpha=model.alpha,
                        aux_weight=model.aux_weight, nonclick_rate=model.nonclick_rate)
                if self.is_cnn_attention:
                    ckpt_data["model_params"] = model.model_params
                if self.logged_pctr_source is not None:
                    ckpt_data["logged_pctr"] = self.logged_pctr_source
                if self.nonclick_sampling is not None:
                    ckpt_data[NONCLICK_SAMPLING_KEY] = self.nonclick_sampling
                torch.save(ckpt_data, checkpoint_path)
                logger.info("  -> checkpoint saved (val_score=%.4f)", score)

            if improved:
                best_score = score
                no_improve = 0
            else:
                no_improve += 1
                if self.es_enabled:
                    logger.info("  no improvement (%d/%d)", no_improve, self.es_patience)
                else:
                    logger.info("  no improvement; early stopping disabled")
                if self.es_enabled and no_improve >= self.es_patience:
                    logger.info("Early stopping triggered after %d epochs with no improvement.", self.es_patience)
                    break

        logger.info("Best val score: %.4f. Loading the %s checkpoint for export...",
                    best_score, "last" if self.keep_checkpoint == "last" else "best")
        ckpt = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        logger.info("%s...", "Exporting predictions" if self.save_predictions else "Scoring splits")
        if use_numpy:
            splits = [("train", dataset.train), ("val", dataset.val), ("test", dataset.test)]
            for split_name, split_data in splits:
                if split_data is None:  # split not loaded (see NumpyDataset.load(load_test=))
                    logger.info("  %s: not loaded, skipping", split_name)
                    continue
                t0 = time.perf_counter()
                ids, ys, zs, scores = self._predict_numpy(model, split_data, split_name=split_name,
                    is_ctr_downsampled_space=downsampling.is_ctr_downsampled_space[split_name])
                path = write_split(self.out_dir, split_name, self.model_name,
                                   ids, ys, zs, scores, self.save_predictions)
                logger.info("  %s: %d rows -> %s (%.1fs)", split_name, len(ids), path, time.perf_counter() - t0)
        else:
            for split_name, imps in [("train", dataset.train), ("val", dataset.val), ("test", dataset.test)]:
                t0 = time.perf_counter()
                if self.model_name == "esmm":
                    ids, ys, zs, qs, vs = self.predict_esmm(
                        model, imps, dataset.vocabs, field_ids,
                        is_ctr_downsampled_space=downsampling.is_ctr_downsampled_space[split_name])
                    scores = {"q": qs, "v": vs}
                else:
                    ids, ys, zs, score = self.predict(model, imps, dataset.vocabs, field_ids)
                    scores = {"score": score}
                path = write_split(self.out_dir, split_name, self.model_name,
                                   ids, ys, zs, scores, self.save_predictions)
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

    def predict_esmm(self, model, impressions: list, vocabs: dict, field_ids: list,
                     *, is_ctr_downsampled_space: bool = False) -> tuple:
        """Return Python-list scores in the explicitly requested population space."""
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
                q = downsampling.model_ctr_for_space(
                    q, self.nonclick_rate, is_ctr_downsampled_space=is_ctr_downsampled_space)
                q_scores.extend(q.cpu().tolist())
                v_scores.extend(v.cpu().tolist())
        return ids, ys, zs, q_scores, v_scores

    # ------------------------------------------------------------------
    # NumpyDataset-backed epoch/predict/val helpers
    # ------------------------------------------------------------------

    def _epoch_order(self, n: int, epoch: int, device: torch.device) -> torch.Tensor:
        """Row order for one epoch, from a generator seeded seed+epoch.

        shuffle_block_size <= 1 permutes rows; larger values permute blocks of
        that many consecutive rows and keep each block's internal order.

        Block shuffling exists to keep reads sequential when a split is streamed
        from disk. This one is resident in device memory and indexed arbitrarily
        at the same cost, so blocks buy nothing here and cost mixing: Ali-CCP is
        ordered by user, and any contiguous 65,536-row run holds only ~1,268
        distinct user bags against ~60,565 for a random draw of that size.
        """
        g = torch.Generator(device=device).manual_seed(self.seed + epoch)
        bs = self.shuffle_block_size
        if bs <= 1:
            return torch.randperm(n, generator=g, device=device)

        n_blocks = (n + bs - 1) // bs
        starts = torch.randperm(n_blocks, generator=g, device=device).to(torch.int64) * bs
        idx = starts.unsqueeze(1) + torch.arange(bs, device=device).unsqueeze(0)
        idx = idx.flatten()
        return idx[idx < n]  # trailing block is short unless n divides evenly

    def _train_epoch_numpy(self, model, split_data, optimizer, epoch: int = 0) -> float:
        """One epoch over a SplitData in block-shuffled order; mean batch loss."""
        model.train()
        n = len(split_data)
        indices = self._epoch_order(n, epoch, split_data.device)
        total_loss, n_batches = 0.0, 0

        for start in range(0, n, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            optimizer.zero_grad()
            loss_val = self._train_step_numpy(model, split_data, batch_idx)
            self._clip_dense_grads(model)
            optimizer.step()
            total_loss += loss_val
            n_batches += 1

            if n_batches % 500 == 0:
                logger.debug("  batch %d/%d  loss=%.4f", n_batches, n // self.batch_size, loss_val)

        return total_loss / n_batches if n_batches else 0.0

    def _clip_dense_grads(self, model) -> None:
        """Clip dense gradients to grad_clip_norm, leaving embeddings untouched.

        Embedding gradients are per-row and sparse in effect; clipping them with
        the dense parameters would scale every touched row by a norm dominated
        by the dense layers.
        """
        if not self.grad_clip_norm:
            return
        embedding_params = {id(p) for p in model.embeddings.parameters()}
        dense = [p for p in model.parameters()
                 if id(p) not in embedding_params and p.grad is not None]
        if dense:
            torch.nn.utils.clip_grad_norm_(dense, self.grad_clip_norm)

    def _train_step_numpy(self, model, split_data, batch_idx: list) -> float:
        """Forward + backward for a NumpyDataset batch.

        No host->device transfer: the split already lives on self.device.
        """
        enc, y, z = split_data.get_batch(batch_idx)
        if self.model_name == "esmm_logged_pctr":
            q, v = model(enc)
            loss = model.loss(
                q, v, y, z, split_data.logged_pctr[batch_idx],
                is_ctr_downsampled_space=downsampling.is_ctr_downsampled_space["train"])
        elif self.model_name == "esmm_direct_product":
            _, v = model(enc)
            loss = model.loss(
                z, split_data.logged_pctr[batch_idx], v)
        elif self.model_name == "esmm":
            from models.esmm import esmm_loss
            q, v = model(enc)
            loss = esmm_loss(y, z, q, v, model.aux_weight)
        elif self.is_cnn_attention:
            from models.cnn_attention import cnn_attention_loss
            click_logits, conversion_logits = model(enc)
            loss = cnn_attention_loss(y, z, click_logits, conversion_logits, model.alpha)
        else:
            from models.dcn import dcn_loss
            loss = dcn_loss(y, model(enc))
        loss.backward()
        return loss.item()

    def _batch_scores(
        self,
        model,       # the model being scored
        split_data,  # the split the rows come from, carrying any logged pCTR
        idx,         # row indices of this batch within that split
        enc,         # those rows' encoded features
        split_name: str,
        *, is_ctr_downsampled_space: bool = False,
    ) -> dict:
        """Named prediction columns for one encoded batch of rows `idx`.

        A single-head model reports one score; ESMM reports its two heads
        separately, leaving any product to the consumer, so that a head can be
        scored on its own population as well as inside a product. An ESMM with no
        auxiliary CTR head has no q. is_ctr_downsampled_space preserves raw CTR when true; split_name only controls logged-score reporting. Logged pCTR is included only for test
        reporting and logged-model validation, never for train reports.
        """
        if self.is_esmm:
            q, v = model(enc)
            q = downsampling.model_ctr_for_space(
                q, self.nonclick_rate, is_ctr_downsampled_space=is_ctr_downsampled_space)
            scores = {"v": v} if q is None else {"q": q, "v": v}
            if split_name in self.logged_ctcvr_splits:
                scores["p"] = split_data.logged_pctr[idx]
            return scores
        if self.is_cnn_attention:
            click_logits, conversion_logits = model(enc)
            return {"q": torch.sigmoid(click_logits), "c": torch.sigmoid(conversion_logits)}
        return {"score": torch.sigmoid(model(enc))}

    def _selection_inputs(
        self,
        scores: dict,  # prediction columns for the validation split
        y,             # click label
        z,             # conversion label; y*z is the CTCVR label
    ) -> tuple:
        """(score, label) pair the configured checkpoint metric is computed on."""
        def column(name):
            if name not in scores:
                # e.g. selecting on the native product with no auxiliary CTR head.
                raise ValueError(f"checkpoint_metric {self.checkpoint_metric!r} needs the {name!r} "
                                 f"score; this model reports {sorted(scores)}")
            return scores[name]

        if self.checkpoint_metric == "pcvr_ne":
            clicked = y == 1
            return column("v")[clicked], z[clicked]
        if self.checkpoint_metric.startswith("cold_ctr_"):
            return column("q_cold"), y
        if self.checkpoint_metric in ("ctr_auc", "ctr_ne", "ctr_prauc"):
            return column("q" if self.is_esmm or self.is_cnn_attention else "score"), y
        if self.checkpoint_metric in ("ctcvr_prauc", "ctcvr_ne"):
            return column("q") * column("v"), y * z
        if self.checkpoint_metric in ("logged_ctcvr_prauc", "logged_ctcvr_ne"):
            if "val" not in self.logged_ctcvr_splits:
                raise ValueError("validation p*v requires esmm_logged_pctr; native ESMM validates on q*v")
            return column("p") * column("v"), y * z
        raise ValueError(f"Unknown checkpoint_metric: {self.checkpoint_metric!r}")

    def _predict_numpy(self, model, split_data, *, split_name: str,
                       is_ctr_downsampled_space: bool = False) -> tuple:
        """Return (sample_ids, ys, zs, scores) from a SplitData, as numpy arrays.

        `scores` is the column -> array mapping `save_scores` writes out.
        is_ctr_downsampled_space selects raw versus recovered CTR independently of split_name.
        """
        model.eval()
        ids_out, ys_out, zs_out = [], [], []
        scores_out = {}
        n = len(split_data)
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                idx = torch.arange(start, min(start + self.batch_size, n),
                                   device=split_data.device)
                enc, y, z = split_data.get_batch(idx)
                ids_out.append(split_data.sample_ids[idx].cpu())
                ys_out.append(y.cpu())
                zs_out.append(z.cpu())
                for column, score in self._batch_scores(model, split_data, idx, enc, split_name,
                                                          is_ctr_downsampled_space=is_ctr_downsampled_space).items():
                    scores_out.setdefault(column, []).append(score.cpu())
        return (
            torch.cat(ids_out).numpy(),
            torch.cat(ys_out).numpy(),
            torch.cat(zs_out).numpy(),
            {c: torch.cat(parts).numpy() for c, parts in scores_out.items()},
        )

    def _val_score_numpy(self, model, val_split) -> tuple[float, dict]:
        """Score ordinary and deterministic cold-start validation populations."""
        from evaluation.metrics import auc, compute_metrics, normalized_entropy, pr_auc
        model.eval()
        columns, ys, zs, attention_parts = {}, [], [], []
        eval_batch = self.eval_batch_size
        n = len(val_split)
        cold_mask = None
        if self.cold_start_validation is not None:
            rng = np.random.default_rng(self.cold_start_validation["seed"])
            cold_mask = torch.from_numpy(
                rng.random(n) < self.cold_start_validation["missing_rate"])

        with torch.no_grad():
            for start in range(0, n, eval_batch):
                stop = min(start + eval_batch, n)
                idx = torch.arange(start, stop, device=val_split.device)
                enc, y, z = val_split.get_batch(idx)
                batch_scores = self._batch_scores(
                    model, val_split, idx, enc, "val",
                    is_ctr_downsampled_space=downsampling.is_ctr_downsampled_space["val"])
                attention = getattr(model, "last_attention_weights", None)
                if self.is_cnn_attention and attention is not None:
                    attention_parts.append(attention.cpu())
                for column, value in batch_scores.items():
                    columns.setdefault(column, []).append(value.cpu())

                if cold_mask is not None:
                    missing = cold_mask[start:stop].to(val_split.device)
                    click_logits, _ = model(
                        enc,
                        forced_missing={
                            self.cold_start_validation["field"]: missing,
                        },
                    )
                    cold_q = torch.sigmoid(click_logits).cpu()
                    # One deployment-like population: ordinary scores on covered
                    # rows and forced-missing scores on the configured fraction.
                    mixed_q = torch.where(missing.cpu(), cold_q, batch_scores["q"].cpu())
                    columns.setdefault("q_cold", []).append(mixed_q)
                ys.append(y.cpu())
                zs.append(z.cpu())

        scores = {column: torch.cat(parts) for column, parts in columns.items()}
        y, z = torch.cat(ys), torch.cat(zs)
        score, label = self._selection_inputs(scores, y, z)
        if self.checkpoint_metric.endswith("_auc") and not (
            (label == 0).any() and (label == 1).any()
        ):
            raise ValueError(
                f"{self.checkpoint_metric} checkpoint selection requires both "
                "click classes in validation")

        extras = {}
        if self.is_esmm or self.is_cnn_attention:
            from scripts.report_metrics import metric_rows
            rows = metric_rows(
                "val",
                self.model_name,
                y.numpy().astype(_LABEL_DTYPES["y"]),
                z.numpy().astype(_LABEL_DTYPES["z"]),
                {column: values.numpy() for column, values in scores.items()},
            )
            extras = {
                f"val_{_HISTORY_POPULATIONS[row['population']]}_{suffix}": row[metric]
                for row in rows
                for metric, suffix in _HISTORY_METRICS.items()
            }
        if "q_cold" in scores:
            cold_metrics = compute_metrics(scores["q_cold"].numpy(), y.numpy())
            extras.update({
                f"val_cold_ctr_{suffix}": cold_metrics[metric]
                for metric, suffix in _HISTORY_METRICS.items()
            })
        if attention_parts:
            weights = torch.cat(attention_parts).clamp_min(torch.finfo(torch.float32).tiny)
            extras.update({
                "val_attention_click_mean": weights[:, 0].mean().item(),
                "val_attention_conversion_mean": weights[:, 1].mean().item(),
                "val_attention_entropy": (-(weights * weights.log()).sum(dim=1).mean().item()),
                "val_attention_min_task_weight": weights.min().item(),
            })

        if self.checkpoint_metric == "pcvr_ne":
            return extras["val_pcvr_ne"], extras
        if self.checkpoint_metric.endswith("_auc"):
            # Multitask reporting already sorted the full population.
            key = f"val_{self.checkpoint_metric}"
            return (
                extras[key] if key in extras
                else auc(score.numpy(), label.numpy())
            ), extras
        if self.checkpoint_metric.endswith("_ne"):
            return normalized_entropy(score.numpy(), label.numpy()), extras
        return pr_auc(score.numpy(), label.numpy()), extras

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
        loss = esmm_loss(y, z, q, v, model.aux_weight)
        loss.backward()
        return loss.item()
