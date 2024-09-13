import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check if Streamlit is installed
try:
    import streamlit
    print("Streamlit is already installed.")
except ImportError:
    print("Streamlit not found. Installing Streamlit...")
    install("streamlit")
