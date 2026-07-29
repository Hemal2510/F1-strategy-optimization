# F1 DQN vs QRL Benchmark Report

## Evaluation protocol

- Both trained policies are evaluated greedily with exploration disabled.
- The same scenario definitions and seed set are reused for both agents.
- The uploaded action mask is applied before every decision.
- Every lap stores the observation, legal actions, six raw Q-values, selected action, reward and race state.
- Statistical comparisons are paired by scenario, seed, track, year and race name.

## Run completeness

```text
agent episodes completed invalid_actions
  dqn       30        30               0
  qrl       30        30               0
```

## Overall metrics

```text
agent            metric   direction  n     mean       std   median   ci_low  ci_high
  dqn      total_reward      higher 30 -43.1756   47.1181 -33.5801 -60.8885 -27.0808
  dqn    final_position       lower 30  3.56667   4.10788        1  2.23333      5.1
  dqn     position_gain      higher 30 -2.56667   4.10788        0     -4.1     -1.2
  dqn   total_race_time       lower 30  5271.04   747.664  4966.44  5017.92  5549.96
  dqn mean_inference_ms       lower 30 0.599991 0.0563422 0.579296 0.580936 0.620795
  dqn  p95_inference_ms       lower 30 0.895506  0.164023  0.85015 0.843179 0.956404
  dqn         pit_count descriptive 30      1.6   1.10172        1  1.23333        2
  qrl      total_reward      higher 30 -51.1573   57.4337 -32.9872 -73.2299 -32.3798
  qrl    final_position       lower 30  3.16667   2.88974        1  2.16667  4.23333
  qrl     position_gain      higher 30 -2.16667   2.88974        0     -3.2     -1.2
  qrl   total_race_time       lower 30  5314.55   917.965  4990.72  5018.41  5658.42
  qrl mean_inference_ms       lower 30  37.1984   1.90599  36.5915  36.5443  37.8796
  qrl  p95_inference_ms       lower 30  44.3085   4.09856  42.4208  42.9891  45.8296
  qrl         pit_count descriptive 30  1.16667  0.791478        1      0.9  1.46667
```

## Track-level metrics

```text
agent       track            metric  n      mean       std   ci_low   ci_high
  dqn      Monaco      total_reward  9  -47.0702   26.8316 -63.3206  -29.6395
  dqn      Monaco    final_position  9   2.33333       2.5        1         4
  dqn      Monaco     position_gain  9  -1.33333       2.5 -3.11111         0
  dqn      Monaco   total_race_time  9   6273.45   525.451  5983.23   6618.33
  dqn      Monaco mean_inference_ms  9  0.582546 0.0357177  0.56307  0.606877
  dqn      Monaco  p95_inference_ms  9  0.799019 0.0670439 0.757635  0.840269
  dqn      Monaco         pit_count  9   2.11111  0.781736  1.66667   2.66667
  dqn       Monza      total_reward  6  -26.0268   37.7897 -54.0207  0.550441
  dqn       Monza    final_position  6   6.66667   4.76095  3.33333        10
  dqn       Monza     position_gain  6  -5.66667   4.76095       -9  -2.33333
  dqn       Monza   total_race_time  6   4539.01   226.625  4388.31   4694.22
  dqn       Monza mean_inference_ms  6  0.658146 0.0856304 0.590553  0.716529
  dqn       Monza  p95_inference_ms  6   1.01379  0.236223 0.852506   1.20699
  dqn       Monza         pit_count  6         2   1.26491  1.16667         3
  dqn Silverstone      total_reward 15  -47.6983    59.327 -79.1257  -20.8771
  dqn Silverstone    final_position 15   3.06667   4.23365  1.26667       5.4
  dqn Silverstone     position_gain 15  -2.06667   4.23365     -4.4 -0.266667
  dqn Silverstone   total_race_time 15    4962.4   64.3118  4929.94   4992.79
  dqn Silverstone mean_inference_ms 15  0.587195 0.0375152 0.569342  0.606151
  dqn Silverstone  p95_inference_ms 15  0.906083   0.14536  0.83952  0.981481
  dqn Silverstone         pit_count 15   1.13333    1.0601 0.666667   1.66667
  qrl      Monaco      total_reward  9  -55.5548   43.6744 -85.1699  -30.5669
  qrl      Monaco    final_position  9   3.33333       3.5        1   5.66667
  qrl      Monaco     position_gain  9  -2.33333       3.5 -4.66667 -0.777778
  qrl      Monaco   total_race_time  9   6412.69   990.336  5879.23    7076.8
  qrl      Monaco mean_inference_ms  9   37.1431    2.1422   36.017   38.5996
  qrl      Monaco  p95_inference_ms  9   44.2243   4.46555  41.7758   47.2592
  qrl      Monaco         pit_count  9   1.11111  0.333333        1   1.33333
  qrl       Monza      total_reward  6  -79.4022   87.4356 -140.251  -19.1213
  qrl       Monza    final_position  6       6.5   1.64317  5.33333   7.66667
  qrl       Monza     position_gain  6      -5.5   1.64317 -6.66667  -4.33333
  qrl       Monza   total_race_time  6   4519.83   180.352  4399.58   4642.04
  qrl       Monza mean_inference_ms  6   37.6925   1.83858  36.4336   39.1052
  qrl       Monza  p95_inference_ms  6   44.8063   3.09363  42.7251   47.2232
  qrl       Monza         pit_count  6         1  0.894427 0.333333   1.66667
  qrl Silverstone      total_reward 15  -37.2209   49.4988 -64.7294  -16.6275
  qrl Silverstone    final_position 15   1.73333   1.53375        1   2.53333
  qrl Silverstone     position_gain 15 -0.733333   1.53375 -1.53333         0
  qrl Silverstone   total_race_time 15   4973.56   54.4851  4946.01   4999.09
  qrl Silverstone mean_inference_ms 15    37.034   1.88677  36.1687   37.9958
  qrl Silverstone  p95_inference_ms 15   44.1599   4.45521  42.2871   46.5654
  qrl Silverstone         pit_count 15   1.26667   0.96115 0.866667   1.73333
```

## Paired DQN–QRL comparison

```text
           metric   direction paired_n   mean_a   mean_b mean_b_minus_a difference_ci_low difference_ci_high paired_effect_size_dz  wilcoxon_p numerical_winner
     total_reward      higher       30 -43.1756 -51.1573       -7.98176          -28.6591            12.7152             -0.135329     0.41613              dqn
   final_position       lower       30  3.56667  3.16667           -0.4              -1.9           0.833333             -0.103256    0.893642              qrl
    position_gain      higher       30 -2.56667 -2.16667            0.4              -0.9            1.86667              0.103256    0.893642              qrl
  total_race_time       lower       30  5271.04  5314.55        43.5149           -37.483            157.245              0.155659    0.416472              dqn
mean_inference_ms       lower       30 0.599991  37.1984        36.5984           35.9732            37.3033               19.3363 1.86265e-09              dqn
 p95_inference_ms       lower       30 0.895506  44.3085         43.413           42.0767            44.9799               10.6335 1.86265e-09              dqn
        pit_count descriptive       30      1.6  1.16667      -0.433333          -1.06667                0.2             -0.244133    0.166217      descriptive
```

## Strategy divergence

```text
scenario seed       track year race_name aligned_steps action_disagreement_rate pit_disagreement_rate pit_lap_jaccard_similarity
 default 1000       Monza 2024       HUL            52                0.0384615             0.0384615                          0
 default 1001 Silverstone 2022       RIC            51                0.0588235             0.0588235                          0
 default 1002 Silverstone 2023       HAM            51                0.0392157             0.0392157                          0
 default 1003 Silverstone 2024       PIA            51                0.0392157             0.0392157                          0
 default 1004      Monaco 2022       LEC            63                 0.031746              0.015873                        0.5
 default 1005      Monaco 2023       ALO            77                 0.038961              0.038961                          0
 default 1006       Monza 2023       PIA            50                     0.06                  0.06                          0
 default 1007      Monaco 2024       PIA            77                 0.038961              0.038961                          0
 default 1008      Monaco 2023       SAI            77                0.0649351             0.0649351                          0
 default 1009      Monaco 2022       LEC            63                 0.031746              0.015873                        0.5
 default 1010      Monaco 2022       SAI            63                 0.031746              0.015873                        0.5
 default 1011 Silverstone 2024       SAI            51                0.0392157             0.0392157                          0
 default 1012      Monaco 2022       RUS            63                 0.031746              0.015873                        0.5
 default 1013       Monza 2024       STR            52                0.0576923             0.0576923                          0
 default 1014 Silverstone 2023       ALB            51                0.0392157             0.0392157                          0
 default 1015 Silverstone 2023       HAM            51                0.0392157             0.0392157                          0
 default 1016 Silverstone 2023       LEC            51                0.0392157             0.0392157                          0
 default 1017 Silverstone 2022       STR            51                0.0588235             0.0588235                          0
 default 1018 Silverstone 2022       RIC            51                0.0588235             0.0588235                          0
 default 1019      Monaco 2023       SAI            77                 0.038961              0.038961                          0
 default 1020       Monza 2022       OCO            52                0.0769231             0.0769231                          0
 default 1021       Monza 2024       PER            52                0.0576923             0.0576923                          0
 default 1022       Monza 2022       LEC            52                0.0576923             0.0576923                          0
 default 1023 Silverstone 2024       STR            51                0.0392157             0.0392157                          0
 default 1024 Silverstone 2023       RUS            51                0.0784314             0.0784314                          0
 default 1025      Monaco 2022       GAS            63                 0.047619              0.047619                          0
 default 1026 Silverstone 2023       PIA            51                0.0392157             0.0392157                          0
 default 1027 Silverstone 2023       VER            51                0.0392157             0.0392157                          0
 default 1028 Silverstone 2023       TSU            51                0.0588235             0.0588235                          0
 default 1029 Silverstone 2023       ZHO            51                0.0392157             0.0392157                          0
```

## Computational profile

```text
agent parameter_count trainable_parameter_count checkpoint_size_mb                                                checkpoint_sha256 mean_inference_ms median_inference_ms p95_inference_ms episode_wall_time_s
  dqn          137607                    137607            2.11988 f8b04bc164909e6de57a38ff4778642b5fe703b3da4aabb6015b2a2cb05b4f02          0.599991            0.545675         0.895506           0.0577386
  qrl             407                       407          0.0214272 a2ba7b2e81908dd27d81899581837c6e95e541a2cecfe4ed32d048ba94cec846           37.1984             36.1505          44.3085             2.24817
```

## Interpretation rules

1. Race outcome metrics such as final position and race time should be interpreted before reward alone.
2. A numerical winner is not automatically a statistically or practically meaningful winner.
3. Use the confidence interval of the paired difference, effect size and per-seed win counts together.
4. Q-values from DQN and QRL may have different scales; compare their chosen actions and action margins rather than treating raw scale equality as required.
5. Inference latency must be reported because the eight-qubit simulated quantum circuit has a different computational profile from the classical network.

## Errors

_No benchmark execution errors were recorded._
