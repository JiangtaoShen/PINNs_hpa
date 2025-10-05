#!/usr/bin/env python3
import subprocess
import sys


def main():
    scripts = [f"m{i}_pde6.py" for i in range(1, 8)]

    for script in scripts:
        print(f"运行 {script}...")
        subprocess.run([sys.executable, script])


if __name__ == "__main__":
    main()