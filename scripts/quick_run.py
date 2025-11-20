import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spot_scam.pipeline.train import run

if __name__ == "__main__":
    print("ABOUT TO CALL RUN")
    run(skip_transformer=True)
