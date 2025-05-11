@echo off
SETLOCAL

echo Setting up package-pricing-ml project...

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Create necessary directories
echo Creating directories...
mkdir data\raw\transactions
mkdir data\raw\behavior
mkdir data\raw\surveys
mkdir data\raw\competitors
mkdir data\processed
mkdir data\external
mkdir data\logs

:: Create sample config
echo Creating sample config...
mkdir config
python src\data\collect_all.py --config config\data_collection.yaml

echo Setup complete!
echo Next steps:
echo 1. Edit config\data_collection.yaml with your actual settings
echo 2. Run 'python src\data\collect_all.py' to collect data
echo 3. Explore the data using notebooks\01_data_exploration.ipynb

ENDLOCAL