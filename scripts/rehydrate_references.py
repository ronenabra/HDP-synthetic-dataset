"""Restore pinned generation inputs before installing from a source checkout."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from synthetic_population.rehydrate import main

if __name__ == "__main__":
    raise SystemExit(main())
