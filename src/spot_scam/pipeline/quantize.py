from __future__ import annotations

import json
from pathlib import Path

import torch
import typer
from transformers import AutoModelForSequenceClassification

from spot_scam.utils.logging import configure_logging
from spot_scam.utils.paths import ARTIFACTS_DIR

logger = configure_logging(__name__)


def main(
    output_dir: Path = typer.Option(
        ARTIFACTS_DIR / "transformer" / "quantized",
        "--output-dir",
        help="Destination for quantized weights.",
    ),
    dtype: str = typer.Option(
        "qint8", "--dtype", help="Target dtype (currently only qint8 supported)."
    ),
) -> None:
    """Quantize the fine-tuned transformer checkpoint using dynamic quantization."""
    best_dir = ARTIFACTS_DIR / "transformer" / "best"
    if not best_dir.exists():
        raise typer.Exit(f"Base checkpoint not found at {best_dir}. Run training first.")

    logger.info("Loading model from %s", best_dir)
    model = AutoModelForSequenceClassification.from_pretrained(best_dir)

    if dtype.lower() != "qint8":  # pragma: no cover
        raise typer.Exit("Only qint8 dynamic quantization is supported at the moment.")

    logger.info("Applying dynamic quantization (torch.nn.Linear -> int8)...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "model.pt"
    torch.save(quantized_model.state_dict(), target_path)
    logger.info("Quantized weights saved to %s", target_path)

    metadata_path = ARTIFACTS_DIR / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    else:
        metadata = {}
    metadata.setdefault("quantized", {})
    metadata["quantized"].update(
        {
            "available": True,
            "dtype": "dynamic_int8",
            "path": str(target_path.relative_to(ARTIFACTS_DIR)),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Metadata updated with quantized model info.")


def entrypoint():
    typer.run(main)


if __name__ == "__main__":
    entrypoint()
