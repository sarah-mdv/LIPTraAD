# LIPTraAD
Chalmers and ETHZ Masters Thesis: Learning Interpretable Prototype Trajectories for Patients with Alzheimer's Disease

## Requirements

- OS: Linux (not tested on other OSs)
- Python 3.7 or higher
- PyTorch version which is compatible with your setup

## Setup

To install the all dependencies needed for this project to run please run the
following commands.
The first command can be skipped if the repository was restored from an
archive.
```shell script
# Clone the project
git clone git@github.com:sarah-mdv/LIPTraAD.git
# Change into project directory
cd LIPTraAD
# Create a virtual environment
python -m venv venv
# Activate it venv
source venv/bin/activate
# Upgrade pip
pip install --upgrade pip
# Install requirements
pip install -r requirements
# Install the project
pip install -e .
# Install PyTorch. Version may be adjusted based on your local setup (GPU, driver versions, etc)
# check: https://pytorch.org/
pip install torch
```
