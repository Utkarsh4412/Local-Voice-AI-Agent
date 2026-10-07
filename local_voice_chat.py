"""
local_voice_chat.py — thin shim for backwards compatibility.

The implementation has moved to the `vaak` package. This file exists so that:
  python local_voice_chat.py [options]
continues to work for anyone using the old entry point.

All logic is in vaak/cli.py. Do not add code here.
"""

from vaak.cli import main

if __name__ == "__main__":
    main()
