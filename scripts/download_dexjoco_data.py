"""Windows/Linux entry point; no torch, numpy, or huggingface_hub needed."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thinker_wm.data_prepare.dexjoco.download import main

if __name__ == "__main__":
    main()
