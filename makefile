clean:
	rm -rf *.o *.out *.exe *.stackdump *.dSYM
	rm -rf build/*
	rm -rf dist/*

format:
	clang-format -i src/_ngram/*.cpp src/_ngram/*.hpp

uninstall: clean
	pip uninstall -y $(PACKAGE_NAME)

install: uninstall
	python setup.py install

demo:
	python demo.py