from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
FIGS_DIR = EXPERIMENTS_DIR / "figs"
TABLES_DIR = EXPERIMENTS_DIR / "tables"
TRACKING_DIR = PROJECT_ROOT / "tracking"
TRACKING_FEEDBACK_DIR = TRACKING_DIR / "feedback"
TRACKING_PREDICTIONS_DIR = TRACKING_DIR / "predictions"
TRACKING_CASES_DIR = TRACKING_DIR / "cases"


def ensure_directories() -> None:
    """Create commonly used directories if they do not yet exist."""
    for directory in [
        CONFIG_DIR,
        DATA_DIR,
        PROCESSED_DIR,
        ARTIFACTS_DIR,
        EXPERIMENTS_DIR,
        FIGS_DIR,
        TABLES_DIR,
        TRACKING_DIR,
        TRACKING_FEEDBACK_DIR,
        TRACKING_PREDICTIONS_DIR,
        TRACKING_CASES_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
