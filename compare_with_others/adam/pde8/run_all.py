#!/usr/bin/env python3
import subprocess
import sys


def main():
    # 依次运行 m1_pde8.py 到 m7_pde8.py
    for i in range(2, 8):
        script = f"m{i}_pde8.py"
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