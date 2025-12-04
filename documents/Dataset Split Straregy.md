# Dataset Split Strategy-----test version

**Total Datasets:** 29  
**Split:** 20 Training (69%) | 9 Testing (31%)

## Overview

This document outlines the strategy for splitting the 29 available datasets into training and testing subsets for process mining heuristic development and ML model evaluation.

## Datasets Available

| # | UUID |
|---|------|
| 1 | `01345ac4-7d1d-426e-92b8-24933a079412` |
| 2 | `12683249` |
| 3 | `2b02709f-9a84-4538-a76a-eb002eacf8d1` |
| 4 | `3301445f-95e8-4ff0-98a4-901f1f204972` |
| 5 | `33632f3c-5c48-40cf-8d8f-2db57f5a6ce7` |
| 6 | `3537c19d-6c64-4b1d-815d-915ab0e479da` |
| 7 | `3926db30-f712-4394-aebc-75976070e91f` |
| 8 | `3cfa2260-f5c5-44be-afe1-b70d35288d6d` |
| 9 | `3d5ae0ce-198c-4b5c-b0f9-60d3035d07bf` |
| 10 | `500573e6-accc-4b0c-9576-aa5468b10cee` |
| 11 | `5f3067df-f10b-45da-b98b-86ae4c7a310b` |
| 12 | `63a8435a-077d-4ece-97cd-2c76d394d99c` |
| 13 | `679b11cf-47cd-459e-a6de-9ca614e25985` |
| 14 | `6a0a26d2-82d0-4018-b1cd-89afb0e8627f` |
| 15 | `6af6d5f0-f44c-49be-aac8-8eaa5fe4f6fd` |
| 16 | `86977bac-f874-49cf-8337-80f26bf5d2ef` |
| 17 | `91fd1fa8-4df4-4b1a-9a3f-0116c412378f` |
| 18 | `9b99a146-51b5-48df-aa70-288a76c82ec4` |
| 19 | `a0addfda-2044-4541-a450-fdcc9fe16d17` |
| 20 | `a6f651a7-5ce0-4bc6-8be1-a7747effa1cc` |
| 21 | `b32c6fe5-f212-4286-9774-58dd53511cf8` |
| 22 | `c2c3b154-ab26-4b31-a0e8-8f2350ddac11` |
| 23 | `c3f3ba2d-e81e-4274-87c7-882fa1dbab0d` |
| 24 | `d06aff4b-79f0-45e6-8ec8-e19730c248f1` |
| 25 | `d9769f3d-0ab0-4fb8-803b-0d1120ffcf54` |
| 26 | `db35afac-2133-40f3-a565-2dc77a9329a3` |
| 27 | `e30ba0c8-0039-4835-a493-6e3aa2301d3f` |
| 28 | `ed445cdd-27d5-4d77-a1f7-59fe7360cfbe` |
| 29 | `fb84cf2d-166f-4de2-87be-62ee317077e5` |

## Split Strategy

### Training Set (20 datasets - 69%)

Use these datasets for model training, feature engineering, and heuristic development:

```
01345ac4-7d1d-426e-92b8-24933a079412
12683249
2b02709f-9a84-4538-a76a-eb002eacf8d1
3301445f-95e8-4ff0-98a4-901f1f204972
33632f3c-5c48-40cf-8d8f-2db57f5a6ce7
3537c19d-6c64-4b1d-815d-915ab0e479da
3926db30-f712-4394-aebc-75976070e91f
3cfa2260-f5c5-44be-afe1-b70d35288d6d
3d5ae0ce-198c-4b5c-b0f9-60d3035d07bf
500573e6-accc-4b0c-9576-aa5468b10cee
5f3067df-f10b-45da-b98b-86ae4c7a310b
63a8435a-077d-4ece-97cd-2c76d394d99c
679b11cf-47cd-459e-a6de-9ca614e25985
6a0a26d2-82d0-4018-b1cd-89afb0e8627f
6af6d5f0-f44c-49be-aac8-8eaa5fe4f6fd
86977bac-f874-49cf-8337-80f26bf5d2ef
91fd1fa8-4df4-4b1a-9a3f-0116c412378f
9b99a146-51b5-48df-aa70-288a76c82ec4
a0addfda-2044-4541-a450-fdcc9fe16d17
c2c3b154-ab26-4b31-a0e8-8f2350ddac11
```

### Testing Set (9 datasets - 31%)

Use these datasets for **unseen evaluation** to validate model generalization:

```
a6f651a7-5ce0-4bc6-8be1-a7747effa1cc
b32c6fe5-f212-4286-9774-58dd53511cf8
c3f3ba2d-e81e-4274-87c7-882fa1dbab0d
d06aff4b-79f0-45e6-8ec8-e19730c248f1
d9769f3d-0ab0-4fb8-803b-0d1120ffcf54
db35afac-2133-40f3-a565-2dc77a9329a3
e30ba0c8-0039-4835-a493-6e3aa2301d3f
ed445cdd-27d5-4d77-a1f7-59fe7360cfbe
fb84cf2d-166f-4de2-87be-62ee317077e5
```