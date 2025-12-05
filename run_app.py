#!/usr/bin/env python3
"""
Launcher script for Interactive Signal Processing Explorer
"""

import subprocess
import sys

def main():
    print("=" * 60)
    print("Interactive Signal Processing Explorer")
    print("=" * 60)
    print("\nStarting Streamlit application...\n")

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have installed all requirements:")
        print("  pip install -r requirements.txt")

if __name__ == "__main__":
    main()
