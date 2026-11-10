## Setup and run

The pinned dependencies are listed in requirements.txt. Use Python 3.10-3.12 (Python 3.11 is recommended). First clone the public repository and enter its folder.

bash
git clone https://github.com/PrathmeshAvadhutJamdade1555/Predict-Energy-Efficiency.git
cd Predict-Energy-Efficiency
python --version
python -m venv .venv


Activate the environment, install the requirements, and run the script:

powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python energy_efficiency.py


bash
# macOS or Linux
source .venv/bin/activate
python -m pip install -r requirements.txt
python energy_efficiency.py


The program writes the dataset exploration log, correlation matrix, and feature-versus-target graph to outputs/data_exploration_and_analysis/. It writes test predictions, evaluation metrics, and the actual-versus-predicted plot to outputs/prediction_results/.****
