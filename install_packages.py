import subprocess
import sys

packages = ['pandas', 'xgboost', 'joblib', 'scikit-learn']
for pkg in packages:
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', pkg], 
                          capture_output=True, text=True)
    print(f"=== Installing {pkg} ===")
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
    print()