PACKAGE_NAME=dnazen

SRC_DIR=src
TEST_DIR=tests

PIXI_SHELL=pixi exec

.git:
	pre-commit install

.PHONY: clean format install test

clean:
	rm -rf *.o *.out *.exe *.stackdump *.dSYM
	rm -rf build/*
	rm -rf dist/*

format:
	clang-format -i src/_ngram/*.cpp src/_ngram/*.hpp
	$(PIXI_SHELL) ruff format $(SRC_DIR)
	$(PIXI_SHELL) ruff format $(TEST_DIR)
	$(PIXI_SHELL) ruff check $(SRC_DIR) --fix
	$(PIXI_SHELL) ruff check $(TEST_DIR) --fix

uninstall: clean
	pip uninstall -y $(PACKAGE_NAME)

install: uninstall .git
	pip install .

test:
	$(MAKE) -C $(TEST_DIR) test