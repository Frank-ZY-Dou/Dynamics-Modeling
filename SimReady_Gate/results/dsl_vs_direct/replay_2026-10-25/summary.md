### claude-haiku-5-5: all tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 48 | 47/48 (89-100%) | 98% | 1.08 | 4020 | 13241 (1600) | $0.002 | 19 s | 2.8 s (2.8 s) | 4106 | 22 s |
| dsl_v0 | 48 | 38/48 (66-88%) | 73% | 1.12 | 6960 | 8914 (3238) | $0.004 | 31 s | 3.1 s (3.1 s) | 8792 | 43 s |
| direct | 48 | 43/48 (78-95%) | 81% | 1.12 | 14374 | 2919 (1352) | $0.007 | 62 s | 0.9 s (0.9 s) | 16046 | 70 s |
| direct+repair | 48 | 37/48 (63-87%) | 77% | 1.12 | 14374 | 2919 (1352) | $0.007 | 62 s | 3.1 s (3.1 s) | 18648 | 84 s |

### claude-haiku-5-5: development tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 42 | 41/42 (88-100%) | 98% | 1.10 | 3925 | 13416 (1617) | $0.002 | 18 s | 2.6 s (2.6 s) | 4020 | 21 s |
| dsl_v0 | 42 | 33/42 (64-88%) | 74% | 1.12 | 6919 | 8562 (2996) | $0.004 | 31 s | 2.8 s (2.8 s) | 8806 | 43 s |
| direct | 42 | 37/42 (75-95%) | 79% | 1.14 | 15992 | 2993 (1333) | $0.008 | 69 s | 0.8 s (0.8 s) | 18153 | 79 s |
| direct+repair | 42 | 31/42 (59-85%) | 74% | 1.14 | 15992 | 2993 (1333) | $0.008 | 69 s | 2.7 s (2.7 s) | 21666 | 96 s |

### claude-haiku-5-5: held-out tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 6 | 6/6 (61-100%) | 100% | 1.00 | 4689 | 12014 (1485) | $0.003 | 21 s | 4.4 s (4.4 s) | 4689 | 26 s |
| dsl_v0 | 6 | 5/6 (44-97%) | 67% | 1.17 | 7249 | 11376 (4932) | $0.005 | 31 s | 4.7 s (4.7 s) | 8698 | 43 s |
| direct | 6 | 6/6 (61-100%) | 100% | 1.00 | 3052 | 2399 (1486) | $0.002 | 14 s | 1.7 s (1.7 s) | 3052 | 15 s |
| direct+repair | 6 | 6/6 (61-100%) | 100% | 1.00 | 3052 | 2399 (1486) | $0.002 | 14 s | 5.6 s (5.6 s) | 3052 | 19 s |

### claude-haiku-5-5: pass rate by task

| task | dsl_v1 | dsl_v0 | direct | direct+repair |
|---|---|---|---|---|
| desk (N=6) | 100% (4396 tok, 23 s) | 67% (9083 tok, 44 s) | 100% (3369 tok, 17 s) | 100% (3369 tok, 20 s) |
| drinks_row (N=6) | 100% (1747 tok, 10 s) | 100% (2315 tok, 11 s) | 67% (1411 tok, 8 s) | 67% (1411 tok, 9 s) |
| drinks_row (N=10) | 67% (2287 tok, 14 s) | 0% (3092 tok, 15 s) | 0% (2078 tok, 10 s) | 0% (2078 tok, 12 s) |
| fruit_cans (N=10) | 100% (3516 tok, 18 s) | 100% (9481 tok, 46 s) | 100% (6616 tok, 28 s) | 100% (6616 tok, 30 s) |
| fruit_cans (N=16) | 100% (5455 tok, 29 s) | 67% (8529 tok, 44 s) | 100% (23038 tok, 100 s) | 67% (23038 tok, 104 s) |
| fruit_cans (N=24) | 100% (4898 tok, 29 s) | 67% (8883 tok, 46 s) | 100% (29007 tok, 123 s) | 100% (29007 tok, 129 s) |
| kitchen_prep (N=12) | 100% (7495 tok, 37 s) | 100% (21449 tok, 96 s) | 100% (11324 tok, 50 s) | 100% (11324 tok, 51 s) |
| minimal_fix (N=12) | 100% (1208 tok, 9 s) | 100% (1062 tok, 9 s) | 100% (10521 tok, 44 s) | 100% (10521 tok, 47 s) |
| pack (N=10) | 100% (2401 tok, 14 s) | 33% (9331 tok, 46 s) | 100% (88352 tok, 377 s) | 0% (88352 tok, 378 s) |
| pack (N=14) | 100% (1091 tok, 8 s) | 33% (7430 tok, 37 s) | 100% (35128 tok, 151 s) | 33% (35128 tok, 152 s) |
| palletize (N=4) | 100% (5258 tok, 25 s) | 100% (2954 tok, 14 s) | 100% (1888 tok, 10 s) | 100% (1888 tok, 10 s) |
| palletize (N=6) | 100% (6261 tok, 31 s) | 100% (4502 tok, 21 s) | 67% (3038 tok, 14 s) | 100% (3038 tok, 15 s) |
| pick_ready (N=8) | 100% (6695 tok, 33 s) | 100% (5006 tok, 25 s) | 100% (4102 tok, 19 s) | 100% (4102 tok, 22 s) |
| setting (N=6) | 100% (3777 tok, 18 s) | 100% (7418 tok, 35 s) | 100% (5599 tok, 26 s) | 100% (5599 tok, 27 s) |
| stack (N=6) | 100% (2854 tok, 16 s) | 100% (5412 tok, 29 s) | 100% (1784 tok, 9 s) | 67% (1784 tok, 11 s) |
| tool_layout (N=7) | 100% (4982 tok, 28 s) | 100% (5414 tok, 28 s) | 100% (2734 tok, 14 s) | 100% (2734 tok, 18 s) |

### claude-haiku-5-5: by scene size

| task | N | dsl_v1 pass, output tokens, model s | direct pass, output tokens, model s |
|---|---|---|---|
| drinks_row | 6 | 3/3, 1747, 9 | 2/3, 1411, 7 |
| drinks_row | 10 | 2/3, 2287, 11 | 0/3, 2078, 10 |
| fruit_cans | 10 | 3/3, 3516, 17 | 3/3, 6616, 28 |
| fruit_cans | 16 | 3/3, 5455, 25 | 3/3, 23038, 99 |
| fruit_cans | 24 | 3/3, 4898, 22 | 3/3, 29007, 122 |
| pack | 10 | 3/3, 2401, 13 | 3/3, 88352, 376 |
| pack | 14 | 3/3, 1091, 6 | 3/3, 35128, 150 |
| palletize | 4 | 3/3, 5258, 24 | 3/3, 1888, 9 |
| palletize | 6 | 3/3, 6261, 29 | 2/3, 3038, 14 |

### claude-opus-5-5: all tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 48 | 48/48 (93-100%) | 100% | 1.02 | 764 | 12414 (1582) | $0.030 | 8 s | 2.6 s (2.6 s) | 764 | 10 s |
| dsl_v0 | 48 | 37/48 (63-87%) | 75% | 1.04 | 2317 | 5891 (1700) | $0.061 | 22 s | 2.4 s (2.4 s) | 3006 | 32 s |
| direct | 48 | 45/48 (83-98%) | 90% | 1.04 | 2481 | 2661 (1665) | $0.063 | 25 s | 0.9 s (0.9 s) | 2646 | 28 s |
| direct+repair | 48 | 39/48 (68-90%) | 81% | 1.04 | 2481 | 2661 (1665) | $0.063 | 25 s | 3.1 s (3.1 s) | 3053 | 35 s |

### claude-opus-5-5: development tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 42 | 42/42 (92-100%) | 100% | 1.02 | 745 | 12471 (1597) | $0.030 | 8 s | 2.4 s (2.4 s) | 745 | 10 s |
| dsl_v0 | 42 | 31/42 (59-85%) | 71% | 1.05 | 2389 | 5955 (1731) | $0.062 | 23 s | 2.1 s (2.1 s) | 3236 | 34 s |
| direct | 42 | 39/42 (81-98%) | 88% | 1.05 | 2676 | 2699 (1691) | $0.067 | 27 s | 0.8 s (0.8 s) | 2882 | 30 s |
| direct+repair | 42 | 33/42 (64-88%) | 79% | 1.05 | 2676 | 2699 (1691) | $0.067 | 27 s | 2.7 s (2.7 s) | 3406 | 38 s |

### claude-opus-5-5: held-out tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 6 | 6/6 (61-100%) | 100% | 1.00 | 897 | 12013 (1484) | $0.032 | 8 s | 4.4 s (4.4 s) | 897 | 12 s |
| dsl_v0 | 6 | 6/6 (61-100%) | 100% | 1.00 | 1818 | 5446 (1484) | $0.049 | 17 s | 4.5 s (4.5 s) | 1818 | 21 s |
| direct | 6 | 6/6 (61-100%) | 100% | 1.00 | 1113 | 2397 (1484) | $0.034 | 13 s | 1.7 s (1.7 s) | 1113 | 14 s |
| direct+repair | 6 | 6/6 (61-100%) | 100% | 1.00 | 1113 | 2397 (1484) | $0.034 | 13 s | 5.5 s (5.5 s) | 1113 | 18 s |

### claude-opus-5-5: pass rate by task

| task | dsl_v1 | dsl_v0 | direct | direct+repair |
|---|---|---|---|---|
| desk (N=6) | 100% (982 tok, 12 s) | 100% (1896 tok, 21 s) | 100% (1197 tok, 15 s) | 100% (1197 tok, 19 s) |
| drinks_row (N=6) | 100% (580 tok, 8 s) | 100% (1098 tok, 11 s) | 100% (470 tok, 6 s) | 100% (470 tok, 7 s) |
| drinks_row (N=10) | 100% (621 tok, 10 s) | 0% (1906 tok, 18 s) | 0% (899 tok, 10 s) | 0% (899 tok, 12 s) |
| fruit_cans (N=10) | 100% (821 tok, 10 s) | 100% (2720 tok, 29 s) | 100% (1208 tok, 12 s) | 100% (1208 tok, 14 s) |
| fruit_cans (N=16) | 100% (955 tok, 14 s) | 100% (3671 tok, 40 s) | 100% (4754 tok, 48 s) | 100% (4754 tok, 52 s) |
| fruit_cans (N=24) | 100% (1015 tok, 16 s) | 100% (5299 tok, 59 s) | 100% (5283 tok, 54 s) | 100% (5283 tok, 59 s) |
| kitchen_prep (N=12) | 100% (1479 tok, 15 s) | 100% (3071 tok, 30 s) | 100% (2200 tok, 22 s) | 100% (2200 tok, 24 s) |
| minimal_fix (N=12) | 100% (206 tok, 6 s) | 100% (690 tok, 10 s) | 100% (1842 tok, 18 s) | 100% (1842 tok, 21 s) |
| pack (N=10) | 100% (217 tok, 4 s) | 0% (2358 tok, 25 s) | 100% (11108 tok, 116 s) | 100% (11108 tok, 117 s) |
| pack (N=14) | 100% (221 tok, 6 s) | 0% (4623 tok, 46 s) | 100% (4387 tok, 44 s) | 0% (4387 tok, 45 s) |
| palletize (N=4) | 100% (917 tok, 10 s) | 100% (1250 tok, 13 s) | 100% (554 tok, 6 s) | 100% (554 tok, 7 s) |
| palletize (N=6) | 100% (1303 tok, 15 s) | 100% (1807 tok, 18 s) | 100% (1500 tok, 16 s) | 100% (1500 tok, 16 s) |
| pick_ready (N=8) | 100% (580 tok, 9 s) | 100% (1813 tok, 21 s) | 100% (1190 tok, 13 s) | 100% (1190 tok, 16 s) |
| setting (N=6) | 100% (840 tok, 9 s) | 33% (1858 tok, 19 s) | 100% (1446 tok, 15 s) | 100% (1446 tok, 16 s) |
| stack (N=6) | 100% (678 tok, 10 s) | 100% (1276 tok, 15 s) | 100% (626 tok, 9 s) | 0% (626 tok, 12 s) |
| tool_layout (N=7) | 100% (811 tok, 13 s) | 100% (1740 tok, 21 s) | 100% (1029 tok, 14 s) | 100% (1029 tok, 18 s) |

### claude-opus-5-5: by scene size

| task | N | dsl_v1 pass, output tokens, model s | direct pass, output tokens, model s |
|---|---|---|---|
| drinks_row | 6 | 3/3, 580, 6 | 3/3, 470, 6 |
| drinks_row | 10 | 3/3, 621, 7 | 0/3, 899, 9 |
| fruit_cans | 10 | 3/3, 821, 8 | 3/3, 1208, 12 |
| fruit_cans | 16 | 3/3, 955, 10 | 3/3, 4754, 46 |
| fruit_cans | 24 | 3/3, 1015, 10 | 3/3, 5283, 53 |
| pack | 10 | 3/3, 217, 3 | 3/3, 11108, 116 |
| pack | 14 | 3/3, 221, 3 | 3/3, 4387, 43 |
| palletize | 4 | 3/3, 917, 10 | 3/3, 554, 6 |
| palletize | 6 | 3/3, 1303, 14 | 3/3, 1500, 15 |

### claude-sonnet-5-5: all tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 48 | 48/48 (93-100%) | 94% | 1.06 | 620 | 12990 (1568) | $0.014 | 5 s | 3.1 s (3.1 s) | 620 | 8 s |
| dsl_v0 | 48 | 30/48 (48-75%) | 60% | 1.06 | 1451 | 6689 (1959) | $0.023 | 10 s | 2.7 s (2.7 s) | 2321 | 21 s |
| direct | 48 | 39/48 (68-90%) | 75% | 1.21 | 3223 | 3200 (1764) | $0.039 | 25 s | 1.0 s (1.0 s) | 3966 | 33 s |
| direct+repair | 48 | 37/48 (63-87%) | 75% | 1.21 | 3223 | 3200 (1764) | $0.039 | 25 s | 3.2 s (3.2 s) | 4181 | 37 s |

### claude-sonnet-5-5: development tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 42 | 42/42 (92-100%) | 98% | 1.02 | 529 | 12508 (1560) | $0.013 | 5 s | 2.3 s (2.3 s) | 529 | 7 s |
| dsl_v0 | 42 | 26/42 (47-75%) | 60% | 1.05 | 1361 | 5952 (1662) | $0.021 | 10 s | 2.3 s (2.3 s) | 2199 | 19 s |
| direct | 42 | 33/42 (64-88%) | 71% | 1.24 | 3546 | 3315 (1804) | $0.043 | 28 s | 0.9 s (0.9 s) | 4513 | 37 s |
| direct+repair | 42 | 31/42 (59-85%) | 71% | 1.24 | 3546 | 3315 (1804) | $0.043 | 28 s | 2.8 s (2.8 s) | 4805 | 42 s |

### claude-sonnet-5-5: held-out tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 6 | 6/6 (61-100%) | 67% | 1.33 | 1260 | 16362 (1623) | $0.021 | 8 s | 8.2 s (8.2 s) | 1260 | 16 s |
| dsl_v0 | 6 | 4/6 (30-90%) | 67% | 1.17 | 2079 | 11850 (4034) | $0.038 | 14 s | 5.3 s (5.3 s) | 3118 | 29 s |
| direct | 6 | 6/6 (61-100%) | 100% | 1.00 | 957 | 2400 (1488) | $0.016 | 8 s | 1.7 s (1.7 s) | 957 | 10 s |
| direct+repair | 6 | 6/6 (61-100%) | 100% | 1.00 | 957 | 2400 (1488) | $0.016 | 8 s | 5.6 s (5.6 s) | 957 | 13 s |

### claude-sonnet-5-5: pass rate by task

| task | dsl_v1 | dsl_v0 | direct | direct+repair |
|---|---|---|---|---|
| desk (N=6) | 100% (1829 tok, 23 s) | 33% (3079 tok, 26 s) | 100% (1004 tok, 10 s) | 100% (1004 tok, 14 s) |
| drinks_row (N=6) | 100% (234 tok, 3 s) | 100% (890 tok, 7 s) | 100% (585 tok, 5 s) | 100% (585 tok, 6 s) |
| drinks_row (N=10) | 100% (221 tok, 5 s) | 0% (1095 tok, 9 s) | 33% (809 tok, 6 s) | 33% (809 tok, 8 s) |
| fruit_cans (N=10) | 100% (351 tok, 4 s) | 100% (1744 tok, 15 s) | 100% (1753 tok, 14 s) | 100% (1753 tok, 15 s) |
| fruit_cans (N=16) | 100% (413 tok, 7 s) | 67% (1571 tok, 16 s) | 100% (4737 tok, 40 s) | 100% (4737 tok, 44 s) |
| fruit_cans (N=24) | 100% (975 tok, 13 s) | 33% (2931 tok, 27 s) | 67% (5875 tok, 49 s) | 67% (5875 tok, 54 s) |
| kitchen_prep (N=12) | 100% (1491 tok, 11 s) | 67% (1897 tok, 16 s) | 67% (2059 tok, 16 s) | 67% (2059 tok, 18 s) |
| minimal_fix (N=12) | 100% (169 tok, 5 s) | 100% (144 tok, 5 s) | 100% (1378 tok, 11 s) | 100% (1378 tok, 14 s) |
| pack (N=10) | 100% (158 tok, 3 s) | 33% (2103 tok, 18 s) | 33% (16408 tok, 131 s) | 33% (16408 tok, 132 s) |
| pack (N=14) | 100% (160 tok, 4 s) | 0% (890 tok, 9 s) | 100% (11491 tok, 90 s) | 33% (11491 tok, 92 s) |
| palletize (N=4) | 100% (764 tok, 7 s) | 100% (746 tok, 6 s) | 100% (595 tok, 5 s) | 100% (595 tok, 6 s) |
| palletize (N=6) | 100% (791 tok, 7 s) | 67% (1095 tok, 8 s) | 67% (926 tok, 8 s) | 67% (926 tok, 8 s) |
| pick_ready (N=8) | 100% (470 tok, 6 s) | 100% (1472 tok, 14 s) | 100% (1103 tok, 10 s) | 100% (1103 tok, 13 s) |
| setting (N=6) | 100% (743 tok, 16 s) | 0% (1619 tok, 13 s) | 67% (1464 tok, 12 s) | 67% (1464 tok, 13 s) |
| stack (N=6) | 100% (462 tok, 6 s) | 100% (856 tok, 8 s) | 67% (463 tok, 5 s) | 67% (463 tok, 7 s) |
| tool_layout (N=7) | 100% (691 tok, 10 s) | 100% (1079 tok, 12 s) | 100% (910 tok, 9 s) | 100% (910 tok, 13 s) |

### claude-sonnet-5-5: by scene size

| task | N | dsl_v1 pass, output tokens, model s | direct pass, output tokens, model s |
|---|---|---|---|
| drinks_row | 6 | 3/3, 234, 2 | 3/3, 585, 4 |
| drinks_row | 10 | 3/3, 221, 2 | 1/3, 809, 5 |
| fruit_cans | 10 | 3/3, 351, 3 | 3/3, 1753, 14 |
| fruit_cans | 16 | 3/3, 413, 3 | 3/3, 4737, 39 |
| fruit_cans | 24 | 3/3, 975, 7 | 2/3, 5875, 47 |
| pack | 10 | 3/3, 158, 2 | 1/3, 16408, 129 |
| pack | 14 | 3/3, 160, 2 | 3/3, 11491, 89 |
| palletize | 4 | 3/3, 764, 6 | 3/3, 595, 5 |
| palletize | 6 | 3/3, 791, 6 | 2/3, 926, 7 |

### Regrade, settle and reuse

| model | condition | pass as run | pass regraded | settles | passes and settles | reuse: new starts pass | reuse: layout spread | reuse: new assets pass |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5 | dsl_v1 | 47/48 | 47/48 | 48/48 | 47/48 | 100% of 235 | 8.8 cm | 100% of 47 |
| claude-haiku-5-5 | dsl_v0 | 38/48 | 38/48 | 47/48 | 37/48 | — | — | — |
| claude-haiku-5-5 | direct | 43/48 | 43/48 | 47/48 | 42/48 | 93% of 215 | 0.0 cm | 65% of 43 |
| claude-opus-5-5 | dsl_v1 | 48/48 | 48/48 | 48/48 | 48/48 | 100% of 240 | 13.9 cm | 100% of 48 |
| claude-opus-5-5 | dsl_v0 | 37/48 | 37/48 | 48/48 | 37/48 | — | — | — |
| claude-opus-5-5 | direct | 45/48 | 45/48 | 48/48 | 45/48 | 93% of 225 | 0.0 cm | 56% of 45 |
| claude-sonnet-5-5 | dsl_v1 | 48/48 | 48/48 | 48/48 | 48/48 | 100% of 240 | 12.9 cm | 100% of 48 |
| claude-sonnet-5-5 | dsl_v0 | 30/48 | 30/48 | 48/48 | 30/48 | — | — | — |
| claude-sonnet-5-5 | direct | 39/48 | 39/48 | 47/48 | 38/48 | 92% of 195 | 0.0 cm | 56% of 39 |
