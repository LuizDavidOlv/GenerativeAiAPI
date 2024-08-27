SHELL := /bin/bash
ROOT_DIR := $(CURDIR)
SHELL_ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

install:
	pip install -r requirements.txt


format:
	cd API && black . && isort .
	cd ../tests && black . && isort .

run:
	python API/main.py