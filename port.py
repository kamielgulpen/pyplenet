import subprocess
import sys

subprocess.Popen([
    sys.executable, '-m', 'mlflow', 'ui',
    '--backend-store-uri', './mlruns',
    '--port', '5000'
])