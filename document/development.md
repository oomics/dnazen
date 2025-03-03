# Development Guide

## library 结构介绍

在 `src`中，有两个文件夹：

- `_ngram`: 有关ngram抽取的cpp代码

- `dnazen`： 所有的python代码

还有一个文件`_ngram.pyi`是由`stubgen`自动生成，用于拼写提示。如果你修改了`_ngram`中的`pybind.cpp`代码，请重新 stubgen。具体请搜索`python stubgen` 和 `pybind11`的文档

### dnazen

在 `dnazen`中，有四个文件夹：

- `data`: mlm dataset和labeled dataset
- `misc`： helper functions
- `model`： DNAZEN的模型
- `ngram`： 有关ngram的代码，这里会利用`pybind11`调用cpp代码

## 开发流程 （python）

如果你只想修改python的代码，那其实没有太多好说的。注意在容易写错的地方加入对应的测试就行，测试应该写在`tests`。添加完新功能后，跑一遍测试。