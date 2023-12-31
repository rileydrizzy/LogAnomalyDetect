echo "Settting up virtual environment..."
python -m venv env
source env/bin/activate
python -m pip install --upgrade pip
echo " Installing dependencies and libaries" 
python -m pip install -r requirements.txt