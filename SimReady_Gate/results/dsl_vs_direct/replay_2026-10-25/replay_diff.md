| model | condition | trials | pass, first run | pass, replay | changed (lost / gained) | rounds kept / called | solver s / trial, first run (CPU) | solver s / trial, replay (CPU) |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-5-5 | dsl_v1 | 48 | 47/48 | 47/48 | 0 / 0 | 49 / 3 | 6.7 (6.7) | 2.8 (2.8) |
| claude-haiku-5-5 | dsl_v0 | 48 | 36/48 | 38/48 | 1 / 3 | 49 / 5 | 7.1 (7.1) | 3.1 (3.1) |
| claude-haiku-5-5 | direct | 48 | 35/48 | 43/48 | 0 / 8 | 51 / 3 | 1.8 (1.8) | 0.9 (0.9) |
| claude-haiku-5-5 | direct+repair | 48 | 36/48 | 37/48 | 1 / 2 | 51 / 3 | 7.0 (7.0) | 3.1 (3.1) |
| claude-sonnet-5-5 | dsl_v1 | 48 | 48/48 | 48/48 | 0 / 0 | 49 / 2 | 7.6 (7.6) | 3.1 (3.1) |
| claude-sonnet-5-5 | dsl_v0 | 48 | 30/48 | 30/48 | 1 / 1 | 50 / 1 | 6.9 (6.9) | 2.7 (2.7) |
| claude-sonnet-5-5 | direct | 48 | 35/48 | 39/48 | 2 / 6 | 50 / 8 | 1.9 (1.9) | 1.0 (1.0) |
| claude-sonnet-5-5 | direct+repair | 48 | 36/48 | 37/48 | 2 / 3 | 50 / 8 | 7.2 (7.2) | 3.2 (3.2) |
| claude-opus-5-5 | dsl_v1 | 48 | 48/48 | 48/48 | 0 / 0 | 49 / 0 | 6.9 (6.9) | 2.6 (2.6) |
| claude-opus-5-5 | dsl_v0 | 48 | 38/48 | 37/48 | 1 / 0 | 50 / 0 | 6.8 (6.8) | 2.4 (2.4) |
| claude-opus-5-5 | direct | 48 | 40/48 | 45/48 | 0 / 5 | 49 / 1 | 1.7 (1.7) | 0.9 (0.9) |
| claude-opus-5-5 | direct+repair | 48 | 39/48 | 39/48 | 0 / 0 | 49 / 1 | 7.0 (7.0) | 3.1 (3.1) |
