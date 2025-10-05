import subprocess
import os


def run_scripts():
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 要运行的脚本列表
    scripts = ["bo_with_kde.py", "bo_without_kde.py"]

    for script in scripts:
        script_path = os.path.join(current_dir, script)
        if os.path.exists(script_path):
            print(f"\n>>> 正在运行: {script}\n")
            # 直接继承 stdout/stderr，不屏蔽子进程的打印
            subprocess.run(["python", script_path], cwd=current_dir)
        else:
            print(f"未找到脚本: {script_path}")


if __name__ == "__main__":
    run_scripts()
