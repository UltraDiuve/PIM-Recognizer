@echo off
python -m pytest --ignore=dumps/ --verbose --cov=src --cov-report term-missing