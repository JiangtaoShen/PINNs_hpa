import subprocess
import sys

scripts = ['m2_pde5.py', 'm3_pde5.py', 'm4_pde5.py', 'm5_pde5.py', 'm6_pde5.py', 'm7_pde5.py']

for script in scripts:
    print(f"Running {script}...")
    subprocess.run([sys.executable, script])