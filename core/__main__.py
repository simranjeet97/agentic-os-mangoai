"""
core/__main__.py — Entry point for `python -m core.shell`

Usage:
  python -m core.shell
  python -m core.shell --user alice
  python -m core.shell --no-stream
"""

from core.shell import main

if __name__ == "__main__":
    main()
