# F1 DQN vs QRL Benchmark Report

## Experiment coverage

- Evaluation episodes: **50**
- Agents: **dqn, qrl**
- Unique seeds: **25**
- Tracks: **Monaco, Monza, Silverstone**

## Overall mean metrics

| metric              |       dqn |       qrl |
|:--------------------|----------:|----------:|
| final_position      |    3.4400 |    4.3200 |
| invalid_action_rate |    0.0000 |    0.0000 |
| mean_inference_ms   |    0.5964 |   42.1662 |
| mean_lap_delta      |    7.5299 |    9.2688 |
| mean_lap_time       |   93.1550 |   94.7620 |
| p95_inference_ms    |    0.9233 |   50.5830 |
| pit_count           |    1.6400 |    2.0400 |
| position_gain       |    4.0000 |    3.1200 |
| total_race_time     | 5350.2689 | 5442.3099 |
| total_reward        |  -47.9888 |  -46.2383 |

## Paired DQN–QRL comparison

| metric              |   paired_runs |    mean_a |    mean_b |   mean_b_minus_a |   paired_t_p_value |   wilcoxon_p_value | numerical_winner   |
|:--------------------|--------------:|----------:|----------:|-----------------:|-------------------:|-------------------:|:-------------------|
| total_reward        |            25 |  -47.9888 |  -46.2383 |           1.7505 |             0.8943 |             0.5077 | qrl                |
| final_position      |            25 |    3.4400 |    4.3200 |           0.8800 |             0.4292 |             0.4401 | dqn                |
| position_gain       |            25 |    4.0000 |    3.1200 |          -0.8800 |             0.4292 |             0.4401 | dqn                |
| total_race_time     |            25 | 5350.2689 | 5442.3099 |          92.0410 |             0.0056 |             0.0172 | dqn                |
| mean_lap_time       |            25 |   93.1550 |   94.7620 |           1.6070 |             0.0045 |             0.0110 | dqn                |
| mean_lap_delta      |            25 |    7.5299 |    9.2688 |           1.7389 |             0.0029 |             0.0042 | dqn                |
| pit_count           |            25 |    1.6400 |    2.0400 |           0.4000 |             0.1155 |             0.0876 | qrl                |
| invalid_action_rate |            25 |    0.0000 |    0.0000 |           0.0000 |           nan      |           nan      | tie                |
| mean_inference_ms   |            25 |    0.5964 |   42.1662 |          41.5697 |             0.0000 |             0.0000 | dqn                |
| p95_inference_ms    |            25 |    0.9233 |   50.5830 |          49.6597 |             0.0000 |             0.0000 | dqn                |

## How to interpret the results

- Higher reward and position gain are normally better.
- Lower final position, race time, lap time and inference time are better.
- Pit count is descriptive; fewer stops are not automatically better.
- A numerical winner only compares sample means.
- A p-value should be read together with effect size and practical race impact.
- Use at least 20–30 paired seeds before making strong conclusions.
- QRL simulator inference time is not the same as real quantum-hardware execution time.