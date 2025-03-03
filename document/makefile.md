# Introduction to Makefiles in this project

## 为什么要用makefile？

makefile最主要的作用是管理来自`scripts`中的复杂脚本依赖 （具体依赖关系请见[这个文档](./pipeline.md)）。在`scripts`中，有些流程是可以并行跑的。在使用makefile之前，我使用的是shell脚本，手动进行拓扑排序后进行并行。然而，当一些流程发生改变后其拓扑关系也可能发生改变，此时我需要再次手动拓扑排序，对开发效率造成了影响。`makefile`自动帮我处理了脚本之间的拓扑关系，我只需要 `make xxx -jn`便可以同时进行`n`个任务。

## 如何使用makefile 跑 `scripts`里的东西

1. `cd scripts`

2. 仿照 `env.sh.sample`写一个脚本用于注册环境变量

3. `source env.sh` 注意是source而不是sh，否则环境变量不会发生改变

4. `make xxx -j3` 这里 `xxx` 是具体任务, `-j3`是最多并行执行3个任务

> 如果你想跑其他`makefile`的任务， 比如`makefile_ngram_merged.mak`，那么就需要`make -f makefile_ngram_merged.mak xxx -j3`， 这里`xxx`是具体任务（一般是`all`）

## makefile 基本语法

```makefile
.PHONY some_file clean

files := file1 file2
some_file: $(files)
	echo "Look at this variable: " $(files)
	touch some_file

file1:
	touch file1
file2:
	touch file2

clean:
	rm -f file1 file2 some_file
```

这是一个简单的makefile脚本。 `make file1`会运行`touch file1`， `make file2`会运行`touch file2`, 而`make some_file`则会运行 `make file1`和`make file2`

`files := file1 file2` 定义了`files`这个变量

`.PHONY some_file clean`告诉makefile `some_file` 和 `clean`不是文件的名字，而是命令。（其实可以不写，但是写了会让你的脚本 less confusing）