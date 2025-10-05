import subprocess
import sys

# 要运行的脚本列表
scripts = [f"m{i}_pde2.py" for i in range(1, 8)]

for script in scripts:
    print(f"\n=== 正在运行 {script} ===\n")
    try:
        # 调用 python 运行子进程
        subprocess.run([sys.executable, script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n!!! 脚本 {script} 运行失败，错误码 {e.returncode} !!!\n")
    except Exception as e:
        print(f"\n!!! 脚本 {script} 出现异常: {e} !!!\n")
