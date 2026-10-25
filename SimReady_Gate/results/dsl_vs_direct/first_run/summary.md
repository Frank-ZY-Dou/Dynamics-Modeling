### claude-haiku-5-5: all tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 48 | 47/48 (89-100%) | 98% | 1.10 | 4051 | 13464 (1640) | $0.002 | 19 s | 6.7 s (6.7 s) | 4137 | 26 s |
| dsl_v0 | 48 | 36/48 (61-85%) | 73% | 1.21 | 7565 | 8868 (2946) | $0.004 | 34 s | 7.1 s (7.1 s) | 10087 | 54 s |
| direct | 48 | 35/48 (59-83%) | 65% | 1.48 | 15425 | 3905 (2059) | $0.008 | 66 s | 1.8 s (1.8 s) | 21154 | 93 s |
| direct+repair | 48 | 36/48 (61-85%) | 75% | 1.48 | 15425 | 3905 (2059) | $0.008 | 66 s | 7.0 s (7.0 s) | 20566 | 98 s |

### claude-haiku-5-5: development tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 42 | 41/42 (88-100%) | 98% | 1.12 | 3960 | 13671 (1662) | $0.002 | 18 s | 5.7 s (5.7 s) | 4056 | 25 s |
| dsl_v0 | 42 | 31/42 (59-85%) | 74% | 1.21 | 7611 | 8510 (2662) | $0.004 | 34 s | 6.3 s (6.3 s) | 10311 | 55 s |
| direct | 42 | 29/42 (54-81%) | 60% | 1.55 | 17192 | 4120 (2141) | $0.009 | 74 s | 1.6 s (1.6 s) | 24899 | 109 s |
| direct+repair | 42 | 30/42 (56-83%) | 71% | 1.55 | 17192 | 4120 (2141) | $0.009 | 74 s | 5.9 s (6.0 s) | 24069 | 112 s |

### claude-haiku-5-5: held-out tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 6 | 6/6 (61-100%) | 100% | 1.00 | 4689 | 12014 (1485) | $0.003 | 21 s | 13.3 s (13.3 s) | 4689 | 34 s |
| dsl_v0 | 6 | 5/6 (44-97%) | 67% | 1.17 | 7249 | 11376 (4932) | $0.005 | 31 s | 13.2 s (13.2 s) | 8698 | 53 s |
| direct | 6 | 6/6 (61-100%) | 100% | 1.00 | 3052 | 2399 (1486) | $0.002 | 14 s | 3.3 s (3.3 s) | 3052 | 17 s |
| direct+repair | 6 | 6/6 (61-100%) | 100% | 1.00 | 3052 | 2399 (1486) | $0.002 | 14 s | 14.5 s (14.5 s) | 3052 | 28 s |

### claude-haiku-5-5: pass rate by task

| task | dsl_v1 | dsl_v0 | direct | direct+repair |
|---|---|---|---|---|
| desk (N=6) | 100% (4396 tok, 33 s) | 67% (9083 tok, 53 s) | 100% (3369 tok, 18 s) | 100% (3369 tok, 28 s) |
| drinks_row (N=6) | 100% (1747 tok, 11 s) | 100% (2315 tok, 12 s) | 67% (1411 tok, 8 s) | 67% (1411 tok, 10 s) |
| drinks_row (N=10) | 67% (2287 tok, 17 s) | 0% (3092 tok, 17 s) | 0% (2078 tok, 11 s) | 0% (2078 tok, 15 s) |
| fruit_cans (N=10) | 100% (3516 tok, 19 s) | 100% (8794 tok, 44 s) | 100% (6616 tok, 29 s) | 100% (6616 tok, 31 s) |
| fruit_cans (N=16) | 100% (5455 tok, 32 s) | 67% (8529 tok, 47 s) | 100% (38715 tok, 168 s) | 100% (38715 tok, 175 s) |
| fruit_cans (N=24) | 100% (4898 tok, 33 s) | 100% (22068 tok, 115 s) | 100% (38720 tok, 167 s) | 100% (38720 tok, 176 s) |
| kitchen_prep (N=12) | 100% (6352 tok, 32 s) | 67% (16030 tok, 74 s) | 100% (11324 tok, 50 s) | 100% (11324 tok, 53 s) |
| minimal_fix (N=12) | 100% (1208 tok, 18 s) | 100% (1062 tok, 18 s) | 100% (10521 tok, 47 s) | 100% (10521 tok, 60 s) |
| pack (N=10) | 100% (2401 tok, 15 s) | 0% (9331 tok, 47 s) | 33% (69651 tok, 295 s) | 0% (69651 tok, 297 s) |
| pack (N=14) | 100% (1091 tok, 10 s) | 33% (7430 tok, 40 s) | 100% (35128 tok, 151 s) | 33% (35128 tok, 154 s) |
| palletize (N=4) | 100% (6894 tok, 34 s) | 100% (7690 tok, 38 s) | 67% (2944 tok, 15 s) | 100% (2944 tok, 16 s) |
| palletize (N=6) | 100% (6261 tok, 33 s) | 100% (4502 tok, 22 s) | 0% (8536 tok, 41 s) | 100% (8536 tok, 43 s) |
| pick_ready (N=8) | 100% (6695 tok, 39 s) | 100% (5006 tok, 30 s) | 100% (4102 tok, 21 s) | 100% (4102 tok, 28 s) |
| setting (N=6) | 100% (3777 tok, 19 s) | 100% (7418 tok, 36 s) | 100% (5599 tok, 26 s) | 100% (5599 tok, 28 s) |
| stack (N=6) | 100% (2854 tok, 26 s) | 67% (3281 tok, 24 s) | 0% (5344 tok, 28 s) | 0% (5344 tok, 30 s) |
| tool_layout (N=7) | 100% (4982 tok, 36 s) | 100% (5414 tok, 36 s) | 100% (2734 tok, 16 s) | 100% (2734 tok, 28 s) |

### claude-haiku-5-5: by scene size

| task | N | dsl_v1 pass, output tokens, model s | direct pass, output tokens, model s |
|---|---|---|---|
| drinks_row | 6 | 3/3, 1747, 9 | 2/3, 1411, 7 |
| drinks_row | 10 | 2/3, 2287, 11 | 0/3, 2078, 10 |
| fruit_cans | 10 | 3/3, 3516, 17 | 3/3, 6616, 28 |
| fruit_cans | 16 | 3/3, 5455, 25 | 3/3, 38715, 165 |
| fruit_cans | 24 | 3/3, 4898, 22 | 3/3, 38720, 164 |
| pack | 10 | 3/3, 2401, 13 | 1/3, 69651, 293 |
| pack | 14 | 3/3, 1091, 6 | 3/3, 35128, 150 |
| palletize | 4 | 3/3, 6894, 32 | 2/3, 2944, 15 |
| palletize | 6 | 3/3, 6261, 29 | 0/3, 8536, 40 |

### claude-opus-5-5: all tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 48 | 48/48 (93-100%) | 100% | 1.02 | 764 | 12414 (1582) | $0.030 | 8 s | 6.9 s (6.9 s) | 764 | 15 s |
| dsl_v0 | 48 | 38/48 (66-88%) | 75% | 1.06 | 2473 | 6018 (1745) | $0.064 | 24 s | 6.8 s (6.8 s) | 3124 | 39 s |
| direct | 48 | 40/48 (70-91%) | 77% | 1.29 | 2783 | 3281 (2012) | $0.072 | 29 s | 1.7 s (1.7 s) | 3340 | 36 s |
| direct+repair | 48 | 39/48 (68-90%) | 81% | 1.29 | 2783 | 3281 (2012) | $0.072 | 29 s | 7.0 s (7.0 s) | 3425 | 44 s |

### claude-opus-5-5: development tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 42 | 42/42 (92-100%) | 100% | 1.02 | 745 | 12471 (1597) | $0.030 | 8 s | 5.9 s (5.9 s) | 745 | 13 s |
| dsl_v0 | 42 | 32/42 (61-87%) | 71% | 1.07 | 2567 | 6100 (1782) | $0.066 | 25 s | 5.7 s (5.7 s) | 3369 | 40 s |
| direct | 42 | 34/42 (67-90%) | 74% | 1.33 | 3021 | 3407 (2088) | $0.077 | 31 s | 1.4 s (1.4 s) | 3732 | 40 s |
| direct+repair | 42 | 33/42 (64-88%) | 79% | 1.33 | 3021 | 3407 (2088) | $0.077 | 31 s | 6.0 s (6.0 s) | 3846 | 47 s |

### claude-opus-5-5: held-out tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 6 | 6/6 (61-100%) | 100% | 1.00 | 897 | 12013 (1484) | $0.032 | 8 s | 14.4 s (14.4 s) | 897 | 22 s |
| dsl_v0 | 6 | 6/6 (61-100%) | 100% | 1.00 | 1818 | 5446 (1484) | $0.049 | 17 s | 14.4 s (14.4 s) | 1818 | 31 s |
| direct | 6 | 6/6 (61-100%) | 100% | 1.00 | 1113 | 2397 (1484) | $0.034 | 13 s | 3.1 s (3.1 s) | 1113 | 16 s |
| direct+repair | 6 | 6/6 (61-100%) | 100% | 1.00 | 1113 | 2397 (1484) | $0.034 | 13 s | 14.2 s (14.2 s) | 1113 | 27 s |

### claude-opus-5-5: pass rate by task

| task | dsl_v1 | dsl_v0 | direct | direct+repair |
|---|---|---|---|---|
| desk (N=6) | 100% (982 tok, 23 s) | 100% (1896 tok, 32 s) | 100% (1197 tok, 16 s) | 100% (1197 tok, 26 s) |
| drinks_row (N=6) | 100% (580 tok, 9 s) | 100% (1098 tok, 12 s) | 100% (470 tok, 6 s) | 100% (470 tok, 8 s) |
| drinks_row (N=10) | 100% (621 tok, 12 s) | 0% (1906 tok, 21 s) | 0% (899 tok, 10 s) | 0% (899 tok, 15 s) |
| fruit_cans (N=10) | 100% (821 tok, 11 s) | 100% (2720 tok, 30 s) | 100% (1208 tok, 12 s) | 100% (1208 tok, 15 s) |
| fruit_cans (N=16) | 100% (955 tok, 17 s) | 100% (3671 tok, 43 s) | 100% (4754 tok, 48 s) | 100% (4754 tok, 55 s) |
| fruit_cans (N=24) | 100% (1015 tok, 20 s) | 100% (5299 tok, 63 s) | 100% (5283 tok, 55 s) | 100% (5283 tok, 65 s) |
| kitchen_prep (N=12) | 100% (1479 tok, 17 s) | 100% (3071 tok, 32 s) | 100% (2200 tok, 23 s) | 100% (2200 tok, 26 s) |
| minimal_fix (N=12) | 100% (206 tok, 16 s) | 100% (690 tok, 21 s) | 100% (1842 tok, 21 s) | 100% (1842 tok, 35 s) |
| pack (N=10) | 100% (217 tok, 6 s) | 33% (4855 tok, 52 s) | 100% (11108 tok, 116 s) | 100% (11108 tok, 119 s) |
| pack (N=14) | 100% (221 tok, 7 s) | 0% (4623 tok, 48 s) | 100% (4387 tok, 44 s) | 0% (4387 tok, 48 s) |
| palletize (N=4) | 100% (917 tok, 11 s) | 100% (1250 tok, 14 s) | 100% (1708 tok, 19 s) | 100% (1708 tok, 20 s) |
| palletize (N=6) | 100% (1303 tok, 17 s) | 100% (1807 tok, 20 s) | 33% (2899 tok, 30 s) | 100% (2899 tok, 32 s) |
| pick_ready (N=8) | 100% (580 tok, 14 s) | 100% (1813 tok, 26 s) | 100% (1190 tok, 14 s) | 100% (1190 tok, 22 s) |
| setting (N=6) | 100% (840 tok, 10 s) | 33% (1858 tok, 20 s) | 100% (1446 tok, 15 s) | 100% (1446 tok, 17 s) |
| stack (N=6) | 100% (678 tok, 21 s) | 100% (1276 tok, 27 s) | 0% (2906 tok, 37 s) | 0% (2906 tok, 40 s) |
| tool_layout (N=7) | 100% (811 tok, 22 s) | 100% (1740 tok, 30 s) | 100% (1029 tok, 15 s) | 100% (1029 tok, 28 s) |

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
| palletize | 4 | 3/3, 917, 10 | 3/3, 1708, 19 |
| palletize | 6 | 3/3, 1303, 14 | 1/3, 2899, 29 |

### claude-sonnet-5-5: all tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 48 | 48/48 (93-100%) | 94% | 1.06 | 633 | 12989 (1655) | $0.014 | 5 s | 7.6 s (7.6 s) | 633 | 13 s |
| dsl_v0 | 48 | 30/48 (48-75%) | 56% | 1.17 | 1563 | 7300 (2197) | $0.025 | 11 s | 6.9 s (6.9 s) | 2501 | 29 s |
| direct | 48 | 35/48 (59-83%) | 56% | 1.60 | 3341 | 4211 (2625) | $0.044 | 26 s | 1.9 s (1.9 s) | 4581 | 39 s |
| direct+repair | 48 | 36/48 (61-85%) | 69% | 1.60 | 3341 | 4211 (2625) | $0.044 | 26 s | 7.2 s (7.2 s) | 4454 | 45 s |

### claude-sonnet-5-5: development tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 42 | 42/42 (92-100%) | 98% | 1.02 | 529 | 12508 (1560) | $0.013 | 5 s | 5.6 s (5.6 s) | 529 | 10 s |
| dsl_v0 | 42 | 26/42 (47-75%) | 55% | 1.17 | 1481 | 6646 (1885) | $0.023 | 11 s | 5.9 s (5.9 s) | 2393 | 27 s |
| direct | 42 | 29/42 (54-81%) | 50% | 1.69 | 3681 | 4470 (2787) | $0.048 | 29 s | 1.7 s (1.7 s) | 5331 | 45 s |
| direct+repair | 42 | 30/42 (56-83%) | 64% | 1.69 | 3681 | 4470 (2787) | $0.048 | 29 s | 6.2 s (6.2 s) | 5154 | 49 s |

### claude-sonnet-5-5: held-out tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 6 | 6/6 (61-100%) | 67% | 1.33 | 1364 | 16359 (2321) | $0.024 | 9 s | 21.7 s (21.7 s) | 1364 | 31 s |
| dsl_v0 | 6 | 4/6 (30-90%) | 67% | 1.17 | 2140 | 11875 (4381) | $0.040 | 14 s | 14.3 s (14.3 s) | 3210 | 43 s |
| direct | 6 | 6/6 (61-100%) | 100% | 1.00 | 957 | 2400 (1488) | $0.016 | 8 s | 3.1 s (3.1 s) | 957 | 11 s |
| direct+repair | 6 | 6/6 (61-100%) | 100% | 1.00 | 957 | 2400 (1488) | $0.016 | 8 s | 13.7 s (13.7 s) | 957 | 22 s |

### claude-sonnet-5-5: pass rate by task

| task | dsl_v1 | dsl_v0 | direct | direct+repair |
|---|---|---|---|---|
| desk (N=6) | 100% (2036 tok, 44 s) | 33% (3201 tok, 37 s) | 100% (1004 tok, 11 s) | 100% (1004 tok, 20 s) |
| drinks_row (N=6) | 100% (234 tok, 4 s) | 100% (890 tok, 8 s) | 100% (585 tok, 5 s) | 100% (585 tok, 7 s) |
| drinks_row (N=10) | 100% (221 tok, 7 s) | 0% (1095 tok, 11 s) | 33% (809 tok, 7 s) | 33% (809 tok, 11 s) |
| fruit_cans (N=10) | 100% (351 tok, 5 s) | 100% (1744 tok, 17 s) | 100% (2464 tok, 20 s) | 100% (2464 tok, 23 s) |
| fruit_cans (N=16) | 100% (413 tok, 10 s) | 67% (1571 tok, 19 s) | 100% (4737 tok, 40 s) | 100% (4737 tok, 47 s) |
| fruit_cans (N=24) | 100% (975 tok, 17 s) | 33% (2931 tok, 32 s) | 100% (6398 tok, 56 s) | 100% (6398 tok, 66 s) |
| kitchen_prep (N=12) | 100% (1491 tok, 13 s) | 100% (2495 tok, 23 s) | 67% (2059 tok, 16 s) | 67% (2059 tok, 20 s) |
| minimal_fix (N=12) | 100% (169 tok, 16 s) | 100% (144 tok, 16 s) | 100% (1378 tok, 14 s) | 100% (1378 tok, 28 s) |
| pack (N=10) | 100% (158 tok, 4 s) | 33% (2103 tok, 19 s) | 67% (16951 tok, 135 s) | 33% (16951 tok, 138 s) |
| pack (N=14) | 100% (160 tok, 5 s) | 0% (890 tok, 11 s) | 100% (5497 tok, 44 s) | 0% (5497 tok, 48 s) |
| palletize (N=4) | 100% (764 tok, 8 s) | 100% (1558 tok, 14 s) | 33% (2250 tok, 19 s) | 100% (2250 tok, 19 s) |
| palletize (N=6) | 100% (791 tok, 8 s) | 67% (1095 tok, 10 s) | 0% (3671 tok, 30 s) | 100% (3671 tok, 31 s) |
| pick_ready (N=8) | 100% (470 tok, 11 s) | 100% (1472 tok, 18 s) | 100% (1103 tok, 12 s) | 100% (1103 tok, 19 s) |
| setting (N=6) | 100% (743 tok, 17 s) | 0% (1619 tok, 14 s) | 67% (1464 tok, 12 s) | 67% (1464 tok, 14 s) |
| stack (N=6) | 100% (462 tok, 16 s) | 67% (1128 tok, 19 s) | 0% (2171 tok, 20 s) | 0% (2171 tok, 22 s) |
| tool_layout (N=7) | 100% (691 tok, 18 s) | 100% (1079 tok, 20 s) | 100% (910 tok, 11 s) | 100% (910 tok, 23 s) |

### claude-sonnet-5-5: by scene size

| task | N | dsl_v1 pass, output tokens, model s | direct pass, output tokens, model s |
|---|---|---|---|
| drinks_row | 6 | 3/3, 234, 2 | 3/3, 585, 4 |
| drinks_row | 10 | 3/3, 221, 2 | 1/3, 809, 5 |
| fruit_cans | 10 | 3/3, 351, 3 | 3/3, 2464, 19 |
| fruit_cans | 16 | 3/3, 413, 3 | 3/3, 4737, 39 |
| fruit_cans | 24 | 3/3, 975, 7 | 3/3, 6398, 52 |
| pack | 10 | 3/3, 158, 2 | 2/3, 16951, 134 |
| pack | 14 | 3/3, 160, 2 | 3/3, 5497, 43 |
| palletize | 4 | 3/3, 764, 6 | 1/3, 2250, 18 |
| palletize | 6 | 3/3, 791, 6 | 0/3, 3671, 28 |

### Regrade, settle and reuse

| model | condition | pass as run | pass regraded | settles | passes and settles | reuse: new starts pass | reuse: layout spread | reuse: new assets pass |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5 | dsl_v1 | 44/48 | 47/48 | 48/48 | 47/48 | 97% of 235 | 8.9 cm | 100% of 47 |
| claude-haiku-5-5 | dsl_v0 | 36/48 | 36/48 | 47/48 | 35/48 | — | — | — |
| claude-haiku-5-5 | direct | 35/48 | 35/48 | 45/48 | 35/48 | 91% of 175 | 0.0 cm | 66% of 35 |
| claude-sonnet-5-5 | dsl_v1 | 45/48 | 48/48 | 48/48 | 48/48 | 94% of 240 | 13.1 cm | 98% of 48 |
| claude-sonnet-5-5 | dsl_v0 | 30/48 | 30/48 | 48/48 | 30/48 | — | — | — |
| claude-sonnet-5-5 | direct | 35/48 | 35/48 | 47/48 | 35/48 | 91% of 175 | 0.0 cm | 46% of 35 |
| claude-opus-5-5 | dsl_v1 | 45/48 | 48/48 | 48/48 | 48/48 | 93% of 240 | 14.3 cm | 100% of 48 |
| claude-opus-5-5 | dsl_v0 | 38/48 | 38/48 | 48/48 | 38/48 | — | — | — |
| claude-opus-5-5 | direct | 40/48 | 40/48 | 47/48 | 40/48 | 92% of 200 | 0.0 cm | 45% of 40 |
