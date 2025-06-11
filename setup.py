import subprocess
import sys

def run(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

# Install Python packages
run(f"{sys.executable} -m pip install --upgrade pip")
run("pip install highway-env")
run("pip install git+https://github.com/eleurent/rl-agents#egg=rl-agents")
run("pip install moviepy -U")
run("pip install imageio_ffmpeg")
run("pip install pyvirtualdisplay")

# Install system packages (only works on Linux and requires sudo)
run("sudo apt-get update")
run("sudo apt-get install -y xvfb ffmpeg")
