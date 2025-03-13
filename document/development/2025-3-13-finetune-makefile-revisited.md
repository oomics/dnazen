# 重新设计finetune的makefile

1. 主要修改：使用环境变量来指定任务类型和子任务

2. 使用方式：

```bash
make -f makefile_finetune.mak all -jn
```
其中，`-j` 指定并行任务数，`n` 指定使用的GPU数量。

3. 支持的环境变量

- `FINETUNE_LEARNING_RATE`：指定学习率。默认为`3e-5`
- `PER_DEVICE_TRAIN_BATCH_SIZE`：指定训练batch size。默认为`8`
- `PER_DEVICE_EVAL_BATCH_SIZE`：指定验证batch size。默认为`32`
- `DNABERT2`：指定是否使用DNABERT2模型。如果需要指定，请设置为`true`，否则设置为`false`。
- `FINETUNE_DATA_DIR`：指定finetune数据目录(GUE数据，注意不要把`GUE`本身包含到路径中，比如不是`/data/user/GUE` 而是`/data/user` )。**必须设置**。
- `FINETUNE_OUT_DIR`：指定finetune输出目录. 默认为`../data/finetune-results`
- `ALL_EMP_TASK_NAMES`：指定EMP任务类型。默认为`H3K14ac H3K4me1 H3K4me2 H3K4me3`
- `ALL_PROM_DETECTION_TASK_NAMES`：指定prom检测任务类型。默认为`prom_300_all prom_core_all`
- `ALL_TRANS_FACTORS_TASK_NAMES`：指定转录因子任务类型。默认为`3 4`
- `ALL_TRANS_FACTORS_MOUSE_TASK_NAMES`：指定转录因子小鼠任务类型。默认为`3 4`




