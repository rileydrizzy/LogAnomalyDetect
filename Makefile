.DEFAULT_GOAL := help

help:
	@echo "    prepare              desc of the command prepare"
	@echo "    install              desc of the command install"


install:
	@echo "Installing..."
	python -m pip install -r requirements.txt
	pre-commit install
	
activate:
	@echo "Activating virtual environment"
	python source env/bin/activate

setup: install activate

precommit:
	@echo "Running precommit on all files"
	pre-commit run --all-files

export:
	@echo "Exporting dependencies to requirements file"
	python -m pip freeze > requirements.txt

force backup: # To push to Github without running precommit
	git commit --no-verify -m "backup"
	git push origin main