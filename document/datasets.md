# Datasets

## [mspecies](https://drive.google.com/file/d/1dSXJfwGpDSJ59ry9KAp8SugQLK35V83f/view)

`DNABERT2` 的预训练数据，大小为30G 有30亿个碱基。

## [GUE]

`DNABERT2`的下游训练数据，大小为313M。

其中老鼠的负样本是合成数据，应该在训练是尽量删除。(我们称之为useful gue data)

## [hg38](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/latest/hg38.fa.gz)

人类基因组数据,约有3.1G

我们会对hg38作反义链处理，处理后的数据有6G, 6亿个碱基。(我们称之为hg38-all)