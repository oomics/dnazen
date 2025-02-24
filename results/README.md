# Results

## 2025-1-23：

See [this doc](./2025-1-23.md) (需要在repo中才能点击超链接)

## 2025-2-6:

See [this doc](./2025-2-6.md) （需要在repo中才能点击超链接）

## 2025-2-17: matching results of frequency based ngram finder

将每条下游任务的数据的标签，以及模型的预测值，以及对应的ngram匹配内容，ngram匹配数量写进表格。

同时，逐渐提高frequency threshold (从5-10)减少ngram，查看匹配的变化。

模型的预测值使用的是2-17前最后一次实验的结果。

## 2025-2-18：make threshold of frequency based ngram finder higher

设置单一 threshold后如果想要剩下几十万ngram，会造成ngram长度为2的占绝大部分。因此我按照长度设置了不同的threshold（长度为6的 threshold=128，5的threashold=256，以此类推，使得ngram长度分布更加balance （len=2有94k len=3有32k len=4有45k len=5有63k len=6有92k）

## 2025-2-19 

使用2-17的分析方式，对hg38, mspecies上按照频率抽取的ngram进行分析。

## 2025-2-21

使用2-17的分析方式，对mspecies上按照pmi抽取的ngram进行分析。

## 2025-2-22: mspecies-phase-2

This is the matching results and meta-datas from DNAZEN(6.1) in dnazen-results.csv table.

在这个实验中，我使用了HG38-all+GUE-all上使用pmi抽取的ngram，在mspecies上进行了第一阶段预训练，然后再在GUE上进行第二阶段预训练。

## 2025-2-23: overfit detection on promoter detection task

Check whether TATA-box makes the model more likely to mark the data as propoter.

发现在promoter-300-all上，总共有5+3个序列在ngram中出现了tata-box。其中，做错的98个例子中，有13个负样本原始序列中含有TATA-box，10个正样本中含有TATA-box。其中负样本中有3个在ngram中也能找到TATA-box；而在做对的1552个例子中，ngram和原始序列均含有tata box的负样本仅有1个。如毛老师预料的，我们的模型会有轻微的overfit，不过，似乎ngram匹配到tata-box的情况并不多。


## 2025-2-24 将mspecies和GUE两个dataset抽取的ngram合并，而不是一起抽取

通过之前的实验我们发现mspecies+GUE上抽取的ngram在GUE上的匹配结果并不佳。因此我尝试将mspecies+GUE和GUE两个dataset分开抽取ngram再合并，发现性能回升。

由于这不算老师交代的任务，我仅在凌晨1点，利用了晚上的8h进行预训练，确保我的实验未造成资源抢占，再在早上finetune。总共预训练了2000个epoch，实际应该训练15000个epoch。如果继续训练，我预计效果可能有更多提升。同时，抽取mspecies和GUE而不是mspecies+GUE和GUE是一个更可靠的方案。

新的merge后的ngram我暂时未作ngram匹配分析。