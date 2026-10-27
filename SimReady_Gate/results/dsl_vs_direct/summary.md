### claude-haiku-5-5: all tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 48 | 45/48 (83-98%) | 90% | 1.19 | 6761 | 17519 (3130) | $0.004 | 31 s | 2.8 s (2.8 s) | 7211 | 36 s |
| dsl_v0 | 48 | 37/48 (63-87%) | 69% | 1.10 | 8834 | 10734 (4691) | $0.005 | 39 s | 2.2 s (2.2 s) | 11460 | 54 s |
| direct | 48 | 42/48 (75-94%) | 81% | 1.15 | 12474 | 4082 (3011) | $0.007 | 54 s | 0.9 s (0.9 s) | 14257 | 63 s |
| direct+repair | 48 | 37/48 (63-87%) | 77% | 1.15 | 12474 | 4082 (3011) | $0.007 | 54 s | 2.7 s (2.7 s) | 16183 | 74 s |
| dsl_v1_rewrites | 48 | 48/48 (93-100%) | 90% | 1.17 | 6446 | 17575 (3062) | $0.004 | 29 s | 4.0 s (4.0 s) | 6446 | 33 s |

### claude-haiku-5-5: development tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 42 | 39/42 (81-98%) | 88% | 1.21 | 7066 | 18082 (3358) | $0.004 | 33 s | 2.7 s (2.7 s) | 7610 | 38 s |
| dsl_v0 | 42 | 31/42 (59-85%) | 64% | 1.12 | 9106 | 9986 (4249) | $0.005 | 40 s | 2.0 s (2.0 s) | 12338 | 57 s |
| direct | 42 | 36/42 (72-93%) | 79% | 1.17 | 13817 | 4315 (3222) | $0.008 | 60 s | 0.8 s (0.8 s) | 16120 | 71 s |
| direct+repair | 42 | 31/42 (59-85%) | 74% | 1.17 | 13817 | 4315 (3222) | $0.008 | 60 s | 2.4 s (2.4 s) | 18720 | 84 s |
| dsl_v1_rewrites | 42 | 42/42 (92-100%) | 88% | 1.19 | 6706 | 18146 (3280) | $0.004 | 31 s | 4.1 s (4.1 s) | 6706 | 35 s |

### claude-haiku-5-5: held-out tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 6 | 6/6 (61-100%) | 100% | 1.00 | 4622 | 13574 (1537) | $0.003 | 21 s | 3.5 s (3.5 s) | 4622 | 25 s |
| dsl_v0 | 6 | 6/6 (61-100%) | 100% | 1.00 | 6923 | 15968 (7784) | $0.005 | 30 s | 3.5 s (3.5 s) | 6923 | 34 s |
| direct | 6 | 6/6 (61-100%) | 100% | 1.00 | 3075 | 2450 (1537) | $0.002 | 14 s | 1.5 s (1.5 s) | 3075 | 15 s |
| direct+repair | 6 | 6/6 (61-100%) | 100% | 1.00 | 3075 | 2450 (1537) | $0.002 | 14 s | 4.8 s (4.8 s) | 3075 | 19 s |
| dsl_v1_rewrites | 6 | 6/6 (61-100%) | 100% | 1.00 | 4622 | 13574 (1537) | $0.003 | 21 s | 3.5 s (3.5 s) | 4622 | 25 s |

### claude-haiku-5-5: pass rate by task

| task | dsl_v1 | dsl_v0 | direct | direct+repair | dsl_v1_rewrites |
|---|---|---|---|---|---|
| desk (N=6) | 100% (4641 tok, 23 s) | 100% (6828 tok, 33 s) | 100% (3565 tok, 17 s) | 100% (3565 tok, 20 s) | 100% (4641 tok, 23 s) |
| drinks_row (N=6) | 100% (2326 tok, 13 s) | 100% (4027 tok, 18 s) | 67% (1984 tok, 10 s) | 67% (1984 tok, 11 s) | 100% (2326 tok, 13 s) |
| drinks_row (N=10) | 100% (2768 tok, 16 s) | 67% (6249 tok, 27 s) | 33% (2492 tok, 12 s) | 33% (2492 tok, 14 s) | 100% (2768 tok, 16 s) |
| fruit_cans (N=10) | 100% (5651 tok, 29 s) | 67% (6702 tok, 31 s) | 100% (6063 tok, 28 s) | 100% (6063 tok, 29 s) | 100% (6124 tok, 33 s) |
| fruit_cans (N=16) | 100% (4644 tok, 24 s) | 67% (11045 tok, 54 s) | 100% (24315 tok, 108 s) | 100% (24315 tok, 112 s) | 100% (4644 tok, 24 s) |
| fruit_cans (N=24) | 100% (6659 tok, 38 s) | 100% (19133 tok, 89 s) | 100% (23577 tok, 103 s) | 100% (23577 tok, 108 s) | 100% (6590 tok, 45 s) |
| kitchen_prep (N=12) | 100% (9557 tok, 45 s) | 33% (15577 tok, 68 s) | 100% (11054 tok, 49 s) | 33% (11054 tok, 50 s) | 100% (9478 tok, 48 s) |
| minimal_fix (N=12) | 100% (950 tok, 7 s) | 100% (1199 tok, 9 s) | 100% (8326 tok, 37 s) | 100% (8326 tok, 39 s) | 100% (950 tok, 7 s) |
| pack (N=10) | 67% (24536 tok, 114 s) | 0% (19929 tok, 94 s) | 67% (47981 tok, 201 s) | 33% (47981 tok, 202 s) | 100% (14831 tok, 70 s) |
| pack (N=14) | 33% (19401 tok, 96 s) | 33% (16072 tok, 73 s) | 67% (48938 tok, 212 s) | 0% (48938 tok, 213 s) | 100% (23598 tok, 115 s) |
| palletize (N=4) | 100% (5235 tok, 25 s) | 100% (6113 tok, 28 s) | 100% (2188 tok, 10 s) | 100% (2188 tok, 11 s) | 100% (5235 tok, 25 s) |
| palletize (N=6) | 100% (4895 tok, 24 s) | 100% (4740 tok, 21 s) | 100% (3649 tok, 16 s) | 100% (3649 tok, 17 s) | 100% (4895 tok, 24 s) |
| pick_ready (N=8) | 100% (3284 tok, 18 s) | 100% (5617 tok, 27 s) | 100% (4798 tok, 23 s) | 100% (4798 tok, 25 s) | 100% (3340 tok, 19 s) |
| setting (N=6) | 100% (6599 tok, 31 s) | 67% (7371 tok, 34 s) | 67% (6171 tok, 28 s) | 67% (6171 tok, 28 s) | 100% (6686 tok, 33 s) |
| stack (N=6) | 100% (2420 tok, 14 s) | 100% (3716 tok, 21 s) | 100% (1905 tok, 10 s) | 100% (1905 tok, 12 s) | 100% (2420 tok, 14 s) |
| tool_layout (N=7) | 100% (4603 tok, 26 s) | 100% (7019 tok, 35 s) | 100% (2585 tok, 14 s) | 100% (2585 tok, 17 s) | 100% (4603 tok, 26 s) |

### claude-haiku-5-5: by scene size

| task | N | dsl_v1 pass, output tokens, model s | direct pass, output tokens, model s |
|---|---|---|---|
| drinks_row | 6 | 3/3, 2326, 12 | 2/3, 1984, 10 |
| drinks_row | 10 | 3/3, 2768, 14 | 1/3, 2492, 11 |
| fruit_cans | 10 | 3/3, 5651, 26 | 3/3, 6063, 27 |
| fruit_cans | 16 | 3/3, 4644, 21 | 3/3, 24315, 107 |
| fruit_cans | 24 | 3/3, 6659, 30 | 3/3, 23577, 102 |
| pack | 10 | 2/3, 24536, 111 | 2/3, 47981, 201 |
| pack | 14 | 1/3, 19401, 92 | 2/3, 48938, 211 |
| palletize | 4 | 3/3, 5235, 25 | 3/3, 2188, 10 |
| palletize | 6 | 3/3, 4895, 23 | 3/3, 3649, 16 |

### claude-opus-5-5: all tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 48 | 48/48 (93-100%) | 100% | 1.00 | 795 | 13749 (2213) | $0.036 | 8 s | 2.2 s (2.2 s) | 795 | 10 s |
| dsl_v0 | 48 | 38/48 (66-88%) | 69% | 1.12 | 2463 | 7016 (2361) | $0.069 | 23 s | 2.1 s (2.1 s) | 3112 | 32 s |
| direct | 48 | 45/48 (83-98%) | 88% | 1.06 | 3122 | 2813 (1881) | $0.078 | 30 s | 0.8 s (0.8 s) | 3330 | 33 s |
| direct+repair | 48 | 42/48 (75-94%) | 85% | 1.06 | 3122 | 2813 (1881) | $0.078 | 30 s | 2.6 s (2.6 s) | 3568 | 38 s |
| dsl_v1_rewrites | 48 | 48/48 (93-100%) | 100% | 1.00 | 795 | 13749 (2213) | $0.036 | 8 s | 2.6 s (2.6 s) | 795 | 10 s |

### claude-opus-5-5: development tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 42 | 42/42 (92-100%) | 100% | 1.00 | 777 | 13774 (2310) | $0.036 | 8 s | 2.0 s (2.0 s) | 777 | 10 s |
| dsl_v0 | 42 | 32/42 (61-87%) | 67% | 1.12 | 2503 | 7012 (2408) | $0.070 | 24 s | 1.9 s (1.9 s) | 3285 | 34 s |
| direct | 42 | 39/42 (81-98%) | 86% | 1.07 | 3406 | 2866 (1931) | $0.084 | 33 s | 0.7 s (0.7 s) | 3668 | 36 s |
| direct+repair | 42 | 36/42 (72-93%) | 83% | 1.07 | 3406 | 2866 (1931) | $0.084 | 33 s | 2.3 s (2.3 s) | 3973 | 41 s |
| dsl_v1_rewrites | 42 | 42/42 (92-100%) | 100% | 1.00 | 777 | 13774 (2310) | $0.036 | 8 s | 2.4 s (2.4 s) | 777 | 10 s |

### claude-opus-5-5: held-out tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 6 | 6/6 (61-100%) | 100% | 1.00 | 926 | 13572 (1535) | $0.033 | 8 s | 3.7 s (3.7 s) | 926 | 12 s |
| dsl_v0 | 6 | 6/6 (61-100%) | 83% | 1.17 | 2186 | 7039 (2026) | $0.061 | 19 s | 3.7 s (3.7 s) | 2186 | 23 s |
| direct | 6 | 6/6 (61-100%) | 100% | 1.00 | 1134 | 2448 (1534) | $0.035 | 12 s | 1.6 s (1.6 s) | 1134 | 14 s |
| direct+repair | 6 | 6/6 (61-100%) | 100% | 1.00 | 1134 | 2448 (1534) | $0.035 | 12 s | 5.1 s (5.1 s) | 1134 | 17 s |
| dsl_v1_rewrites | 6 | 6/6 (61-100%) | 100% | 1.00 | 926 | 13572 (1535) | $0.033 | 8 s | 4.2 s (4.2 s) | 926 | 13 s |

### claude-opus-5-5: pass rate by task

| task | dsl_v1 | dsl_v0 | direct | direct+repair | dsl_v1_rewrites |
|---|---|---|---|---|---|
| desk (N=6) | 100% (1145 tok, 13 s) | 100% (1896 tok, 19 s) | 100% (1236 tok, 14 s) | 100% (1236 tok, 17 s) | 100% (1145 tok, 13 s) |
| drinks_row (N=6) | 100% (498 tok, 6 s) | 100% (1227 tok, 12 s) | 100% (448 tok, 5 s) | 100% (448 tok, 6 s) | 100% (498 tok, 7 s) |
| drinks_row (N=10) | 100% (682 tok, 10 s) | 0% (1347 tok, 14 s) | 0% (1066 tok, 11 s) | 0% (1066 tok, 13 s) | 100% (682 tok, 11 s) |
| fruit_cans (N=10) | 100% (875 tok, 12 s) | 100% (3628 tok, 35 s) | 100% (2088 tok, 21 s) | 100% (2088 tok, 22 s) | 100% (875 tok, 13 s) |
| fruit_cans (N=16) | 100% (1073 tok, 14 s) | 67% (3922 tok, 41 s) | 100% (6472 tok, 63 s) | 100% (6472 tok, 67 s) | 100% (1073 tok, 15 s) |
| fruit_cans (N=24) | 100% (1111 tok, 15 s) | 100% (6826 tok, 69 s) | 100% (6231 tok, 63 s) | 100% (6231 tok, 68 s) | 100% (1111 tok, 16 s) |
| kitchen_prep (N=12) | 100% (1482 tok, 15 s) | 100% (3913 tok, 39 s) | 100% (1812 tok, 18 s) | 100% (1812 tok, 20 s) | 100% (1482 tok, 15 s) |
| minimal_fix (N=12) | 100% (184 tok, 5 s) | 100% (435 tok, 8 s) | 100% (1723 tok, 17 s) | 100% (1723 tok, 19 s) | 100% (184 tok, 5 s) |
| pack (N=10) | 100% (487 tok, 6 s) | 33% (2666 tok, 28 s) | 100% (17098 tok, 168 s) | 100% (17098 tok, 169 s) | 100% (487 tok, 6 s) |
| pack (N=14) | 100% (493 tok, 7 s) | 33% (2813 tok, 30 s) | 100% (5009 tok, 49 s) | 0% (5009 tok, 51 s) | 100% (493 tok, 8 s) |
| palletize (N=4) | 100% (1018 tok, 10 s) | 100% (1551 tok, 15 s) | 100% (812 tok, 8 s) | 100% (812 tok, 8 s) | 100% (1018 tok, 10 s) |
| palletize (N=6) | 100% (1126 tok, 13 s) | 100% (1791 tok, 17 s) | 100% (1424 tok, 13 s) | 100% (1424 tok, 14 s) | 100% (1126 tok, 13 s) |
| pick_ready (N=8) | 100% (567 tok, 8 s) | 100% (1832 tok, 19 s) | 100% (1378 tok, 14 s) | 100% (1378 tok, 17 s) | 100% (567 tok, 9 s) |
| setting (N=6) | 100% (851 tok, 8 s) | 33% (1701 tok, 17 s) | 100% (1636 tok, 17 s) | 100% (1636 tok, 17 s) | 100% (851 tok, 8 s) |
| stack (N=6) | 100% (426 tok, 7 s) | 100% (1389 tok, 17 s) | 100% (482 tok, 5 s) | 100% (482 tok, 6 s) | 100% (426 tok, 7 s) |
| tool_layout (N=7) | 100% (706 tok, 11 s) | 100% (2477 tok, 26 s) | 100% (1033 tok, 13 s) | 100% (1033 tok, 17 s) | 100% (706 tok, 12 s) |

### claude-opus-5-5: by scene size

| task | N | dsl_v1 pass, output tokens, model s | direct pass, output tokens, model s |
|---|---|---|---|
| drinks_row | 6 | 3/3, 498, 5 | 3/3, 448, 5 |
| drinks_row | 10 | 3/3, 682, 8 | 0/3, 1066, 10 |
| fruit_cans | 10 | 3/3, 875, 11 | 3/3, 2088, 20 |
| fruit_cans | 16 | 3/3, 1073, 10 | 3/3, 6472, 62 |
| fruit_cans | 24 | 3/3, 1111, 10 | 3/3, 6231, 62 |
| pack | 10 | 3/3, 487, 5 | 3/3, 17098, 167 |
| pack | 14 | 3/3, 493, 5 | 3/3, 5009, 49 |
| palletize | 4 | 3/3, 1018, 10 | 3/3, 812, 8 |
| palletize | 6 | 3/3, 1126, 12 | 3/3, 1424, 13 |

### claude-sonnet-5-5: all tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 48 | 48/48 (93-100%) | 90% | 1.12 | 812 | 15533 (2493) | $0.019 | 7 s | 2.6 s (2.6 s) | 812 | 10 s |
| dsl_v0 | 48 | 31/48 (50-77%) | 50% | 1.19 | 1619 | 7231 (2307) | $0.026 | 13 s | 2.2 s (2.2 s) | 2507 | 24 s |
| direct | 48 | 41/48 (73-93%) | 79% | 1.10 | 2134 | 2923 (1934) | $0.029 | 18 s | 0.8 s (0.8 s) | 2498 | 22 s |
| direct+repair | 48 | 36/48 (61-85%) | 75% | 1.10 | 2134 | 2923 (1934) | $0.029 | 18 s | 2.8 s (2.8 s) | 2845 | 28 s |
| dsl_v1_rewrites | 48 | 48/48 (93-100%) | 90% | 1.15 | 859 | 15922 (2595) | $0.020 | 8 s | 5.4 s (5.5 s) | 859 | 13 s |

### claude-sonnet-5-5: development tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 42 | 42/42 (92-100%) | 95% | 1.05 | 674 | 14429 (2392) | $0.018 | 6 s | 2.0 s (2.0 s) | 674 | 8 s |
| dsl_v0 | 42 | 26/42 (47-75%) | 48% | 1.19 | 1633 | 7279 (2369) | $0.026 | 14 s | 2.0 s (2.0 s) | 2638 | 25 s |
| direct | 42 | 36/42 (72-93%) | 79% | 1.12 | 2300 | 2994 (1994) | $0.031 | 19 s | 0.7 s (0.7 s) | 2683 | 23 s |
| direct+repair | 42 | 31/42 (59-85%) | 74% | 1.12 | 2300 | 2994 (1994) | $0.031 | 19 s | 2.4 s (2.4 s) | 3116 | 30 s |
| dsl_v1_rewrites | 42 | 42/42 (92-100%) | 95% | 1.05 | 664 | 14438 (2360) | $0.017 | 6 s | 2.4 s (2.4 s) | 664 | 9 s |

### claude-sonnet-5-5: held-out tasks

| condition | trials | pass (final, 95% CI) | pass (round 1) | rounds | output tokens / trial | prompt tokens / trial (fresh) | cost / trial | model time / trial | solver time / trial (CPU) | output tokens / pass | time / pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsl_v1 | 6 | 6/6 (61-100%) | 50% | 1.67 | 1783 | 23259 (3197) | $0.033 | 15 s | 7.0 s (7.0 s) | 1783 | 22 s |
| dsl_v0 | 6 | 5/6 (44-97%) | 67% | 1.17 | 1519 | 6893 (1880) | $0.023 | 11 s | 3.7 s (3.7 s) | 1823 | 18 s |
| direct | 6 | 5/6 (44-97%) | 83% | 1.00 | 974 | 2429 (1516) | $0.016 | 9 s | 1.6 s (1.6 s) | 1168 | 13 s |
| direct+repair | 6 | 5/6 (44-97%) | 83% | 1.00 | 974 | 2429 (1516) | $0.016 | 9 s | 5.1 s (5.1 s) | 1168 | 17 s |
| dsl_v1_rewrites | 6 | 6/6 (61-100%) | 50% | 1.83 | 2224 | 26312 (4244) | $0.041 | 18 s | 26.8 s (26.9 s) | 2224 | 45 s |

### claude-sonnet-5-5: pass rate by task

| task | dsl_v1 | dsl_v0 | direct | direct+repair | dsl_v1_rewrites |
|---|---|---|---|---|---|
| desk (N=6) | 100% (2025 tok, 26 s) | 67% (1405 tok, 13 s) | 100% (1040 tok, 12 s) | 100% (1040 tok, 15 s) | 100% (2872 tok, 65 s) |
| drinks_row (N=6) | 100% (483 tok, 5 s) | 100% (818 tok, 7 s) | 100% (571 tok, 8 s) | 100% (571 tok, 9 s) | 100% (483 tok, 6 s) |
| drinks_row (N=10) | 100% (536 tok, 8 s) | 0% (1504 tok, 12 s) | 33% (868 tok, 9 s) | 33% (868 tok, 11 s) | 100% (536 tok, 9 s) |
| fruit_cans (N=10) | 100% (400 tok, 5 s) | 100% (1121 tok, 10 s) | 100% (1267 tok, 12 s) | 100% (1267 tok, 13 s) | 100% (400 tok, 5 s) |
| fruit_cans (N=16) | 100% (880 tok, 10 s) | 33% (2635 tok, 25 s) | 67% (5017 tok, 42 s) | 67% (5017 tok, 45 s) | 100% (880 tok, 11 s) |
| fruit_cans (N=24) | 100% (1011 tok, 13 s) | 33% (2541 tok, 26 s) | 100% (4243 tok, 35 s) | 100% (4243 tok, 40 s) | 100% (1011 tok, 14 s) |
| kitchen_prep (N=12) | 100% (1652 tok, 13 s) | 100% (2440 tok, 22 s) | 100% (1852 tok, 15 s) | 100% (1852 tok, 16 s) | 100% (1652 tok, 13 s) |
| minimal_fix (N=12) | 100% (252 tok, 7 s) | 100% (178 tok, 6 s) | 100% (1473 tok, 14 s) | 100% (1473 tok, 16 s) | 100% (252 tok, 7 s) |
| pack (N=10) | 100% (161 tok, 4 s) | 0% (2519 tok, 26 s) | 67% (8164 tok, 70 s) | 0% (8164 tok, 71 s) | 100% (161 tok, 4 s) |
| pack (N=14) | 100% (402 tok, 12 s) | 33% (2346 tok, 21 s) | 100% (3968 tok, 33 s) | 0% (3968 tok, 34 s) | 100% (265 tok, 11 s) |
| palletize (N=4) | 100% (1041 tok, 8 s) | 100% (706 tok, 6 s) | 100% (618 tok, 6 s) | 100% (618 tok, 6 s) | 100% (1041 tok, 8 s) |
| palletize (N=6) | 100% (946 tok, 8 s) | 100% (1068 tok, 10 s) | 100% (982 tok, 10 s) | 100% (982 tok, 10 s) | 100% (946 tok, 8 s) |
| pick_ready (N=8) | 100% (471 tok, 8 s) | 67% (2236 tok, 19 s) | 67% (1176 tok, 12 s) | 67% (1176 tok, 14 s) | 100% (471 tok, 9 s) |
| setting (N=6) | 100% (711 tok, 6 s) | 0% (1464 tok, 12 s) | 67% (1442 tok, 11 s) | 67% (1442 tok, 12 s) | 100% (711 tok, 7 s) |
| stack (N=6) | 100% (484 tok, 7 s) | 100% (1290 tok, 16 s) | 100% (558 tok, 6 s) | 100% (558 tok, 8 s) | 100% (484 tok, 8 s) |
| tool_layout (N=7) | 100% (1540 tok, 19 s) | 100% (1633 tok, 16 s) | 67% (907 tok, 10 s) | 67% (907 tok, 13 s) | 100% (1575 tok, 24 s) |

### claude-sonnet-5-5: by scene size

| task | N | dsl_v1 pass, output tokens, model s | direct pass, output tokens, model s |
|---|---|---|---|
| drinks_row | 6 | 3/3, 483, 4 | 3/3, 571, 8 |
| drinks_row | 10 | 3/3, 536, 6 | 1/3, 868, 8 |
| fruit_cans | 10 | 3/3, 400, 4 | 3/3, 1267, 11 |
| fruit_cans | 16 | 3/3, 880, 7 | 2/3, 5017, 41 |
| fruit_cans | 24 | 3/3, 1011, 8 | 3/3, 4243, 34 |
| pack | 10 | 3/3, 161, 3 | 2/3, 8164, 69 |
| pack | 14 | 3/3, 402, 9 | 3/3, 3968, 32 |
| palletize | 4 | 3/3, 1041, 8 | 3/3, 618, 5 |
| palletize | 6 | 3/3, 946, 7 | 3/3, 982, 10 |

### Regrade, settle and reuse

| model | condition | pass as run | pass regraded | settles | passes and settles | reuse: new starts pass | reuse: layout spread | reuse: new assets pass |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5 | dsl_v1 | 45/48 | 45/48 | 48/48 | 45/48 | 99% of 225 | 10.0 cm | 100% of 45 |
| claude-haiku-5-5 | dsl_v0 | 37/48 | 37/48 | 48/48 | 37/48 | — | — | — |
| claude-haiku-5-5 | direct | 42/48 | 42/48 | 48/48 | 42/48 | 93% of 210 | 0.0 cm | 62% of 42 |
| claude-opus-5-5 | dsl_v1 | 48/48 | 48/48 | 48/48 | 48/48 | 98% of 240 | 13.8 cm | 100% of 48 |
| claude-opus-5-5 | dsl_v0 | 38/48 | 38/48 | 48/48 | 38/48 | — | — | — |
| claude-opus-5-5 | direct | 45/48 | 45/48 | 48/48 | 45/48 | 93% of 225 | 0.0 cm | 67% of 45 |
| claude-sonnet-5-5 | dsl_v1 | 48/48 | 48/48 | 48/48 | 48/48 | 100% of 240 | 13.0 cm | 98% of 48 |
| claude-sonnet-5-5 | dsl_v0 | 31/48 | 31/48 | 48/48 | 31/48 | — | — | — |
| claude-sonnet-5-5 | direct | 41/48 | 41/48 | 48/48 | 41/48 | 93% of 205 | 0.0 cm | 56% of 41 |

### After a failed first answer: dsl_v1 against dsl_v1_rewrites

| model | first answer | pairs | pass, plain / rewrites | gate after round 2 | request weakened | rounds | output tokens after round 1 | plain only / rewrites only | p |
|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5 | conflict | 18 | 18/18 / 18/18 | 16 / 18 | 0 / 0 | 2.11 / 2.00 | 6817 / 4735 | 0 / 0 | 1 |
| claude-haiku-5-5 | margin | 7 | 7/7 / 7/7 | 6 / 6 | 0 / 0 | 2.14 / 2.14 | 11792 / 3423 | 0 / 0 | 1 |
| claude-haiku-5-5 | model | 8 | 5/8 / 8/8 | 7 / 8 | 2 / 0 | 2.12 / 2.00 | 14464 / 12574 | 0 / 3 | 0.25 |
| claude-haiku-5-5 | target | 27 | 27/27 / 27/27 | 26 / 26 | 0 / 0 | 2.04 / 2.04 | 4507 / 4170 | 0 / 0 | 1 |
| claude-haiku-5-5 | tight | 40 | 40/40 / 40/40 | 36 / 37 | 0 / 0 | 2.10 / 2.08 | 5548 / 4917 | 0 / 0 | 1 |
| claude-haiku-5-5 | all | 100 | 97/100 / 100/100 | 91 / 95 | 2 / 0 | 2.09 / 2.05 | 6646 / 5191 | 0 / 3 | 0.25 |
| claude-opus-5-5 | conflict | 18 | 18/18 / 18/18 | 18 / 18 | 0 / 0 | 2.00 / 2.00 | 1033 / 1061 | 0 / 0 | 1 |
| claude-opus-5-5 | margin | 7 | 7/7 / 7/7 | 6 / 7 | 0 / 0 | 2.14 / 2.00 | 2560 / 580 | 0 / 0 | 1 |
| claude-opus-5-5 | target | 27 | 27/27 / 27/27 | 27 / 27 | 0 / 0 | 2.00 / 2.00 | 856 / 844 | 0 / 0 | 1 |
| claude-opus-5-5 | tight | 40 | 40/40 / 40/40 | 40 / 40 | 0 / 0 | 2.00 / 2.00 | 886 / 918 | 0 / 0 | 1 |
| claude-opus-5-5 | all | 92 | 92/92 / 92/92 | 91 / 92 | 0 / 0 | 2.01 / 2.00 | 1033 / 898 | 0 / 0 | 1 |
| claude-sonnet-5-5 | conflict | 18 | 17/18 / 18/18 | 16 / 18 | 1 / 0 | 2.11 / 2.00 | 1248 / 1024 | 0 / 1 | 1 |
| claude-sonnet-5-5 | margin | 7 | 7/7 / 7/7 | 6 / 6 | 0 / 0 | 2.14 / 2.14 | 1298 / 569 | 0 / 0 | 1 |
| claude-sonnet-5-5 | model | 6 | 6/6 / 6/6 | 6 / 5 | 0 / 0 | 2.00 / 2.17 | 968 / 1341 | 0 / 0 | 1 |
| claude-sonnet-5-5 | target | 27 | 27/27 / 27/27 | 27 / 27 | 0 / 0 | 2.00 / 2.00 | 801 / 793 | 0 / 0 | 1 |
| claude-sonnet-5-5 | tight | 40 | 40/40 / 40/40 | 40 / 39 | 0 / 0 | 2.00 / 2.02 | 925 / 957 | 0 / 0 | 1 |
| claude-sonnet-5-5 | all | 98 | 97/98 / 98/98 | 95 / 95 | 1 / 0 | 2.03 / 2.03 | 979 / 920 | 0 / 1 | 1 |
