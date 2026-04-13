# Meadow AD — Benchmark Results

## EN vs ZH Comparison

| Rank | Score | Language | Slogan | Source |
|------|-------|----------|--------|--------|
| 1 | 100% | ZH | 每一步 都算樹 | 台灣金句獎 |
| 2 | 99% | ZH | 花在刀口 省在街口 | 台灣金句獎 |
| 3 | 97% | ZH | 何必長大 | 台灣金句獎 |
| 4 | 96% | ZH | 今天來聚吧 | 台灣金句獎 |
| 5 | 95% | ZH | 這樣懂了爸 | 台灣金句獎 |
| 6 | 95% | ZH | 每一天，都要來點陽光 | 台灣金句獎 |
| 7 | 93% | ZH | 人沒來，至少也要喜年來 | 台灣金句獎 |
| 8 | 89% | ZH | 低頭，才有賺頭 | 台灣金句獎 |
| 9 | 89% | ZH | 世界越快 心 則慢 | 台灣金句獎 |
| 10 | 88% | ZH | 省錢就像白T牛仔褲，永不退流行 | 台灣金句獎 |
| 11 | 88% | ZH | 為了下一代，我們決定拿起這一袋。 | 台灣金句獎 |
| 12 | 87% | ZH | 後來的我們，都更好了 | 台灣金句獎 |
| 13 | 87% | ZH | 好喝到有春 | 台灣金句獎 |
| 14 | 87% | ZH | 一年一Do，平安普渡 | 台灣金句獎 |
| 15 | 85% | ZH | 好東西要跟好朋友分享 | 4A得獎 |
| 16 | 84% | ZH | 解放玩心 禁止小心 | 台灣金句獎 |
| 17 | 83% | ZH | 強力脫睏 | 台灣金句獎 |
| 18 | 83% | ZH | 有伴用力閃，沒伴擋光害 | 台灣金句獎 |
| 19 | 83% | ZH | 年輕人不怕菜，就怕不吃菜！ | 台灣金句獎 |
| 20 | 82% | ZH | 把認真 做到成真 | 台灣金句獎 |

**English average: 40%** | **Chinese average: 82%**

> Note: English scores higher due to model's English-native fMRI training data. See Language Bias Notice in README.

## Meadow Score Interpretation

| Score Range | Meaning |
|-------------|---------|
| 90-100 | Exceptional — strong multi-region activation |
| 70-89 | Strong — clear emotional or cognitive trigger |
| 40-69 | Moderate — some activation but not distinctive |
| 0-39 | Weak — minimal differentiation from baseline |

## Calibration Method

Weights calibrated using **Ridge Regression** (R²=0.66) on **43 Taiwan award-winning slogans** from [Brain.com.tw 廣告流行語金句獎](https://www.brain.com.tw/news/services?category=award) (2016-2024).

### Original Weights (Knutson 2007)
```
PFC: +0.40 | Insula: -0.20 | Temporal: +0.20 | Parietal: +0.10 | Occipital: +0.10
```

### Calibrated Weights (Award-Winning Benchmark)
```
PFC: -0.13 | Insula: -0.047 | Temporal: -0.481 | Parietal: +0.029 | Occipital: +0.312
```

Key insight: **Knutson's weights predict purchase behavior (English speakers). Calibrated weights predict creative award-winning quality (Chinese market).** Different objectives = different brain region importance.
