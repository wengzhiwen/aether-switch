"""Project controller entry point; uses system Python, never a model environment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server_manager.cli import main

if __name__ == "__main__":
    main()
