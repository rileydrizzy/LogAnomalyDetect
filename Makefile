
.DEFAULT_GOAL := help

help:
	@echo "    prepare              desc of the command prepare"
	@echo "    install              desc of the command install"

install: #int
	@echo "Installing..."
	poetry install
	poetry run pre-commit install
	
activate:
	@echo "Activating virtual environment"
	poetry shell

setup: install activate

precommit: #
	@echo "Running precommit on all files"
	pre-commit run --all-files