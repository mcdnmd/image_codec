build_cpp:
	@CFLAGS='-std=c++11' python3 EntropySetup.py build_ext --inplace


start:
	@poetry run python3 -m solution


test:
	@poetry run pytest -vvv tests