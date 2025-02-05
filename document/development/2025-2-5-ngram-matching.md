## Ngram matching statistics

Calculate the average number of ngram matches for right/wrong classification, and true/false positive/negative.

### General Results: Average number of ngram matches for right/wrong classification

|    | Right | Wrong |
|----|-------|-------|
|   H3K4me1    |0.7018|0.7160|
|   H3K4me2    |0.7084|0.7121|
|   H3K4me3    |0.7337|0.6907|
|   H3K79me3   |0.6887|0.6748|
| prom_300_all |0.8580|1.2452|
| prom_core_all|0.3650|0.3874|
|     tf_3     |1.3414|0.5000|
|   mouse_4    |0.3204|0.2870|

### Confusion Matrix

The average number of ngram matches and the number of true/false positive/negative in confusion matrix.

|          |pred_pos|pred_neg|
|----------| -------|--------|
|actual_pos| avg(count) | - |
|actual_neg| - | - |


#### H3K4me1
|pred_pos|pred_neg|
| -------|--------|
| 0.7371 | 0.6944 |
| 0.7444 | 0.6771 |

#### H3K4me2
|pred_pos|pred_neg|
| -------|--------|
| 0.6775 | 0.6892 |
| 0.7280 | 0.7307 |

#### H3K4me3
|pred_pos|pred_neg|
| -------|--------|
| 0.7246 | 0.7066 |
| 0.6841 | 0.7450 |

#### H3K79me3
|pred_pos|pred_neg|
|-|-|
| 0.7084 | 0.5805 |
| 0.7133 | 0.6889 |

#### prom_300_all
|pred_pos|pred_neg|
|-|-|
| 0.6331(2786) | 1.5407(154) |
| 0.9765(209) | 1.0687(2771) |

#### prom_core_all

|pred_pos|pred_neg|
|-|-|
|0.3592(2478) | 0.3734(474) |
|0.3989(574) | 0.3709(2394) |

#### tf_3
|pred_pos|pred_neg|
|-|-|
| 2.5487(359) | 0.3546(141) |
| 0.7887(71)  | 0.3310(429) |
* Actual neg match should be 0.3867

#### mouse_4

|pred_pos|pred_neg|
|-|-|
| 0.2959(686) | 0.3047(256) |
| 0.2647(204) | 0.3433(737) |