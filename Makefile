build_cpp:
	@CFLAGS='-std=c++11' python3 ./solution/entropy_codec/EntropySetup.py build_ext --inplace

start:
	@poetry run python3 -m solution $(model_type)


test:
	@poetry run pytest -vvv tests


lint:
	@ruff check solution tests

format:
	@ruff format solution tests
