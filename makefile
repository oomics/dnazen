PACKAGE_NAME=dnazen

SRC_DIR=src
TEST_DIR=tests

.git:
	pre-commit install

.PHONY: clean format install test

clean:
	rm -rf *.o *.out *.exe *.stackdump *.dSYM
	rm -rf build/*
	rm -rf dist/*

format:
	clang-format -i src/_ngram/*.cpp src/_ngram/*.hpp
	ruff format
	ruff check --fix

uninstall: clean
	pip uninstall -y $(PACKAGE_NAME)

install: uninstall .git
	pip install .

test:
	$(MAKE) -C $(TEST_DIR) test