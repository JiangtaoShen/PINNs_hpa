#!/usr/bin/env python3
import subprocess
import sys


def main():
    for i in range(2, 8):
        script = f"m{i}_pde3.py"
        print(f"运行 {script}...")

        try:
            result = subprocess.run([sys.executable, script], check=True)
            print(f"✅ {script} 运行完成\n")
        except subprocess.CalledProcessError:
            print(f"❌ {script} 运行失败\n")
        except FileNotFoundError:
            print(f"❌ {script} 文件不存在\n")


if __name__ == "__main__":
    main()