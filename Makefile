
.DEFAULT_GOAL := help

help:
	@echo "    prepare              desc of the command prepare"
	@echo "    install              desc of the command install"

install: 
	@echo "Installing..."
	poetry install
	poetry run pre-commit install
	
activate:
	@echo "Activating virtual environment"
	poetry shell

setup: install activate
