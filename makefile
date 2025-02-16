PACKAGE_NAME=dnazen
CPP_MODULE_NAME=_ngram

SRC_DIR=src
TEST_DIR=tests

STUBGEN_OUT=$(SRC_DIR)/$(CPP_MODULE_NAME).pyi

PIXI_SHELL=pixi exec

.git:
	pre-commit install

.PHONY: clean format install test

clean:
	rm -rf *.o *.out *.exe *.stackdump *.dSYM
	rm -rf build/*
	rm -rf dist/*

format:
	$(PIXI_SHELL) clang-format -i src/_ngram/*.cpp src/_ngram/*.hpp
	$(PIXI_SHELL) ruff format $(SRC_DIR)
	$(PIXI_SHELL) ruff format $(TEST_DIR)
	$(PIXI_SHELL) ruff check $(SRC_DIR) --fix
	$(PIXI_SHELL) ruff check $(TEST_DIR) --fix

$(STUBGEN_OUT):
	pybind11-stubgen $(CPP_MODULE_NAME)
	mv stubs/$(CPP_MODULE_NAME).pyi $(STUBGEN_OUT)

stubgen: $(STUBGEN_OUT)

uninstall: clean
	pip uninstall -y $(PACKAGE_NAME)

install: uninstall .git $(STUBGEN_OUT)
	pip install .

test:
	$(MAKE) -C $(TEST_DIR) test