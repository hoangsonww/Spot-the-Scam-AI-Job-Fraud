from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_callback import EarlyStoppingCallback

from spot_scam.evaluation.metrics import MetricResults, compute_metrics, optimal_threshold
from spot_scam.utils.logging import configure_logging

logger = configure_logging(__name__)


@dataclass
class TransformerRun:
    name: str
    model_dir: Path
    tokenizer_dir: Path
    val_scores: np.ndarray
    test_scores: np.ndarray
    test_labels: np.ndarray
    val_metrics: MetricResults
    threshold: float
    train_time: float


def train_transformer_model(
    train_df,
    val_df,
    test_df,
    config: Dict,
    output_dir: Path,
) -> TransformerRun:
    """
    Fine-tune a compact transformer (DistilBERT by default) on the `text_all` feature.
    """
    transformer_conf = config["models"]["transformer"]
    project_conf = config["project"]
    set_seed(project_conf["random_seed"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_name = transformer_conf["model_name"]
    learning_rate = float(transformer_conf["learning_rate"])
    weight_decay = float(transformer_conf["weight_decay"])
    batch_size = int(transformer_conf["batch_size"])
    num_epochs = int(transformer_conf["epochs"])
    gradient_accumulation = int(transformer_conf.get("gradient_accumulation", 1))
    warmup_ratio = float(transformer_conf.get("warmup_ratio", 0.1))
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_func(batch):
        return tokenizer(
            batch["text_all"],
            truncation=True,
            padding="max_length",
            max_length=transformer_conf["max_length"],
        )

    train_dataset = Dataset.from_pandas(
        train_df[["text_all", config["data"]["target_column"]]].rename(
            columns={config["data"]["target_column"]: "labels"}
        ),
        preserve_index=False,
    )
    val_dataset = Dataset.from_pandas(
        val_df[["text_all", config["data"]["target_column"]]].rename(
            columns={config["data"]["target_column"]: "labels"}
        ),
        preserve_index=False,
    )
    test_dataset = Dataset.from_pandas(
        test_df[["text_all", config["data"]["target_column"]]].rename(
            columns={config["data"]["target_column"]: "labels"}
        ),
        preserve_index=False,
    )

    train_dataset = train_dataset.map(tokenize_func, batched=True)
    val_dataset = val_dataset.map(tokenize_func, batched=True)
    test_dataset = test_dataset.map(tokenize_func, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    num_labels = len(np.unique(train_df[config["data"]["target_column"]]))
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Detect platform for FP16 compatibility
    import platform
    use_fp16 = transformer_conf.get("fp16", False)
    if use_fp16 and platform.system() == "Darwin":
        logger.warning("Disabling FP16 on macOS due to MPS backend limitations")
        use_fp16 = False

    training_args_kwargs = {
        "output_dir": str(output_dir / "transformer"),
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": num_epochs,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "logging_steps": 10,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "seed": project_conf["random_seed"],
        "fp16": use_fp16,
        "max_grad_norm": transformer_conf.get("max_grad_norm", 1.0),
        "dataloader_num_workers": config["project"].get("num_workers", 2),
        "gradient_accumulation_steps": gradient_accumulation,
        "report_to": "none",
        "save_strategy": "epoch",
        "overwrite_output_dir": True,
    }
    # Hugging Face 4.57 renamed evaluation_strategy -> eval_strategy.
    if "eval_strategy" in TrainingArguments.__init__.__code__.co_varnames:
        training_args_kwargs["eval_strategy"] = "epoch"
    else:  # pragma: no cover - backwards compatibility
        training_args_kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**training_args_kwargs)

    def compute_eval_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        pred_labels = (probs >= 0.5).astype(int)
        precision, recall, f1, _ = _precision_recall_f1(labels, pred_labels)
        return {"precision": precision, "recall": recall, "f1": f1}

    callbacks = []
    early_stopping_patience = transformer_conf.get("early_stopping_patience")
    if early_stopping_patience is not None and early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(early_stopping_patience),
                early_stopping_threshold=0.0,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_eval_metrics,
        callbacks=callbacks,
    )

    logger.info("Starting transformer fine-tuning for %d epochs (max).", transformer_conf["epochs"])
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time

    val_predictions = trainer.predict(val_dataset)
    val_probs = torch.softmax(torch.tensor(val_predictions.predictions), dim=1)[:, 1].numpy()
    test_predictions = trainer.predict(test_dataset)
    test_probs = torch.softmax(torch.tensor(test_predictions.predictions), dim=1)[:, 1].numpy()

    threshold = optimal_threshold(
        val_predictions.label_ids,
        val_probs,
        metric=config["evaluation"]["thresholds"]["optimize_metric"],
    )

    val_metrics = compute_metrics(
        val_predictions.label_ids,
        val_probs,
        metrics_list=config["evaluation"]["metrics"],
        threshold=threshold,
        positive_label=1,
    )

    model_dir = output_dir / "transformer" / "best"
    tokenizer_dir = output_dir / "transformer" / "tokenizer"
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(tokenizer_dir))

    logger.info(
        "Transformer validation F1=%.3f Precision=%.3f Recall=%.3f",
        val_metrics.values.get("f1", np.nan),
        val_metrics.values.get("precision", np.nan),
        val_metrics.values.get("recall", np.nan),
    )

    return TransformerRun(
        name=model_name,
        model_dir=model_dir,
        tokenizer_dir=tokenizer_dir,
        val_scores=val_probs,
        test_scores=test_probs,
        test_labels=test_predictions.label_ids,
        val_metrics=val_metrics,
        threshold=threshold,
        train_time=train_time,
    )


def _precision_recall_f1(y_true, y_pred):
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return precision, recall, f1, _


__all__ = ["train_transformer_model", "TransformerRun"]
