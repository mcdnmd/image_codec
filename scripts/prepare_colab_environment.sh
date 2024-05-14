#!/bin/sh

set -e

pip install --upgrade pip
pip install poetry==1.8.2
poetry config virtualenvs.create false
poetry install

make build_cpp