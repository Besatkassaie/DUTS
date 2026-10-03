
**Pure Starmie (no fairification) -- results**

| bench | k | reached τ | precision | recall | ideal recall | mean ΣU | mean F |
|---|---|---:|---:|---:|---:|---:|---:|
| santos3 | 5 | 21/48 | 0.888 | 0.2772 | 0.2984 | 60.0 | 0.337 |
| santos3 | 10 | 12/48 | 0.829 | 0.4216 | 0.4736 | 115.7 | 0.269 |
| santos3 | 20 | 9/48 | 0.752 | 0.6516 | 0.7299 | 220.0 | 0.206 |
| tusSmall3 | 5 | 7/92 | 0.900 | 0.0098 | 0.0105 | 50.2 | 0.161 |
| tusSmall3 | 10 | 5/92 | 0.886 | 0.0195 | 0.0211 | 100.1 | 0.126 |
| tusSmall3 | 20 | 1/92 | 0.843 | 0.0358 | 0.0421 | 198.5 | 0.094 |
| tusLarge3 | 5 | 22/142 | 0.792 | 0.0099 | 0.0125 | 55.9 | 0.183 |
| tusLarge3 | 10 | 13/142 | 0.769 | 0.0191 | 0.0249 | 111.0 | 0.143 |
| tusLarge3 | 20 | 8/142 | 0.713 | 0.0348 | 0.0498 | 219.4 | 0.108 |

**Pure Starmie (no fairification) -- algorithm details**

| bench | k | mean candidates scored | queries with < k returned |
|---|---|---:|---:|
| santos3 | 5 | 105.8 | 0 |
| santos3 | 10 | 105.8 | 0 |
| santos3 | 20 | 105.8 | 0 |
| tusSmall3 | 5 | 144.4 | 0 |
| tusSmall3 | 10 | 144.4 | 0 |
| tusSmall3 | 20 | 144.4 | 0 |
| tusLarge3 | 5 | 126.5 | 0 |
| tusLarge3 | 10 | 126.5 | 0 |
| tusLarge3 | 20 | 126.5 | 0 |

**Exhaustive swap (no filtering) -- results**

| bench | k | feasible | precision | recall | ideal recall | mean ΣU | mean F_R |
|---|---|---:|---:|---:|---:|---:|---:|
| santos3 | 5 | 47/48 | 0.875 | 0.2502 | 0.2984 | 55.7 | 0.397 |
| santos3 | 10 | 42/48 | 0.704 | 0.3113 | 0.4736 | 93.9 | 0.338 |
| santos3 | 20 | 31/48 | 0.441 | 0.3340 | 0.7299 | 106.4 | 0.242 |
| tusSmall3 | 5 | 66/92 | 0.676 | 0.0085 | 0.0105 | 29.6 | 0.239 |
| tusSmall3 | 10 | 43/92 | 0.429 | 0.0135 | 0.0211 | 32.8 | 0.151 |
| tusSmall3 | 20 | 24/92 | 0.227 | 0.0202 | 0.0421 | 29.4 | 0.081 |
| tusLarge3 | 5 | 108/142 | 0.628 | 0.0086 | 0.0125 | 37.4 | 0.264 |
| tusLarge3 | 10 | 77/142 | 0.399 | 0.0129 | 0.0249 | 44.8 | 0.181 |
| tusLarge3 | 20 | 49/142 | 0.205 | 0.0166 | 0.0498 | 44.7 | 0.111 |

**Exhaustive swap (no filtering) -- algorithm details**

| bench | k | mean candidates scored | queries with < k returned | needed fairification | swap succeeded (of needed) | mean swaps (of needed) | mean F before swap (of needed) |
|---|---|---:|---:|---:|---:|---:|---:|
| santos3 | 5 | 106.2 | 1 | 28/48 | 27/28 | 1.6 | 0.221 |
| santos3 | 10 | 106.2 | 6 | 36/48 | 30/36 | 3.4 | 0.170 |
| santos3 | 20 | 106.2 | 17 | 39/48 | 22/39 | 5.3 | 0.131 |
| tusSmall3 | 5 | 144.9 | 26 | 85/92 | 59/85 | 2.4 | 0.144 |
| tusSmall3 | 10 | 144.9 | 49 | 87/92 | 38/87 | 3.6 | 0.115 |
| tusSmall3 | 20 | 144.9 | 68 | 90/92 | 22/90 | 3.7 | 0.089 |
| tusLarge3 | 5 | 125.5 | 34 | 122/142 | 88/122 | 2.2 | 0.150 |
| tusLarge3 | 10 | 125.5 | 65 | 129/142 | 64/129 | 3.4 | 0.120 |
| tusLarge3 | 20 | 125.5 | 93 | 134/142 | 41/134 | 3.6 | 0.093 |

**Nested-loop swap (winnow filtering) -- results**

| bench | k | feasible | precision | recall | ideal recall | mean ΣU | mean F_R |
|---|---|---:|---:|---:|---:|---:|---:|
| santos3 | 5 | 47/48 | 0.846 | 0.2277 | 0.2984 | 55.2 | 0.392 |
| santos3 | 10 | 35/48 | 0.612 | 0.2495 | 0.4736 | 83.3 | 0.292 |
| santos3 | 20 | 20/48 | 0.317 | 0.2348 | 0.7299 | 68.9 | 0.170 |
| tusSmall3 | 5 | 52/92 | 0.528 | 0.0076 | 0.0105 | 20.1 | 0.189 |
| tusSmall3 | 10 | 33/92 | 0.320 | 0.0116 | 0.0211 | 22.3 | 0.114 |
| tusSmall3 | 20 | 12/92 | 0.105 | 0.0095 | 0.0421 | 13.7 | 0.040 |
| tusLarge3 | 5 | 101/142 | 0.569 | 0.0083 | 0.0125 | 32.1 | 0.245 |
| tusLarge3 | 10 | 59/142 | 0.284 | 0.0109 | 0.0249 | 28.7 | 0.141 |
| tusLarge3 | 20 | 39/142 | 0.150 | 0.0134 | 0.0498 | 32.2 | 0.090 |

**Nested-loop swap (winnow filtering) -- algorithm details**

| bench | k | mean candidates scored | queries with < k returned | needed fairification | swap succeeded (of needed) | mean swaps (of needed) | mean F before swap (of needed) | dominated removed / pool | removed share | mean candidates after winnow |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| santos3 | 5 | 106.1 | 0 | 28/48 | 27/28 | 1.8 | 0.221 | 1876/2102 | 0.892 | 6.2 |
| santos3 | 10 | 106.1 | 0 | 36/48 | 23/36 | 3.2 | 0.170 | 2406/2673 | 0.900 | 4.2 |
| santos3 | 20 | 106.1 | 0 | 39/48 | 11/39 | 3.2 | 0.131 | 2330/2593 | 0.899 | 3.5 |
| tusSmall3 | 5 | 144.7 | 0 | 85/92 | 45/85 | 2.2 | 0.145 | 10828/11306 | 0.958 | 3.4 |
| tusSmall3 | 10 | 144.7 | 0 | 87/92 | 28/87 | 2.8 | 0.114 | 10709/11184 | 0.958 | 2.7 |
| tusSmall3 | 20 | 144.7 | 0 | 90/92 | 10/90 | 3.0 | 0.088 | 10619/11077 | 0.959 | 2.1 |
| tusLarge3 | 5 | 125.1 | 0 | 121/142 | 80/121 | 2.2 | 0.149 | 12042/13031 | 0.924 | 6.0 |
| tusLarge3 | 10 | 125.1 | 0 | 128/142 | 45/128 | 3.2 | 0.122 | 12371/13443 | 0.920 | 5.2 |
| tusLarge3 | 20 | 125.1 | 0 | 134/142 | 31/134 | 3.3 | 0.094 | 12585/13694 | 0.919 | 5.0 |
