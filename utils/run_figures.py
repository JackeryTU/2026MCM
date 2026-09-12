import subprocess, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
for script in ["fig_overview_q1.py","fig_q2.py","fig_q3.py","fig_q4_validation.py"]:
    subprocess.run([sys.executable,str(HERE/script)],check=True)
subprocess.run([sys.executable,str(HERE/"apply_selected_downloads.py")],check=True)
print("All publication figures generated.")
