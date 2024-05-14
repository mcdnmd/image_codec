#!/bin/sh

set -e

pip install --upgrade pip
pip install poetry==1.8.2
poetry install

make build_cpp