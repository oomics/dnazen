# Results

## 2025-2-22: mspecies-phase-2

This is the matching results and meta-datas from DNAZEN(6.1) in dnazen-results.csv table.

在这个实验中，我使用了HG38-all+GUE-all上使用pmi抽取的ngram，在mspecies上进行了第一阶段预训练，然后再在GUE上进行第二阶段预训练。

## 2025-2-24 将mspecies和GUE两个dataset抽取的ngram合并，而不是一起抽取

通过之前的实验我们发现mspecies+GUE上抽取的ngram在GUE上的匹配结果并不佳。因此我尝试将mspecies+GUE和GUE两个dataset分开抽取ngram再合并，发现性能回升。

由于这不算老师交代的任务，我仅在凌晨1点，利用了晚上的8h进行预训练，确保我的实验未造成资源抢占，再在早上finetune。总共预训练了2000个epoch，实际应该训练15000个epoch。如果继续训练，我预计效果可能有更多提升。同时，抽取mspecies和GUE而不是mspecies+GUE和GUE是一个更可靠的方案。

新的merge后的ngram我暂时未作ngram匹配分析。