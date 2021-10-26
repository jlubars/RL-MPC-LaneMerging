These are the configuration files used for all the experiments in our paper. There are a few different types of configurations:

1. `st_{traffic_type}.json`

This evaluates the ST solver on the specified traffic type.

2. `train_{traffic_type}_{seed}.json`

This trains a DDPG model with the corresponding traffic type and seed, then evaluates its performance.

3. `combined_{traffic_type}_{seed}.json`

This evaluates the combination of the ST solver and the DDPG trained with the specified traffic type and seed. You need to run `train_{traffic_type}_{seed}` with the same traffic type and seed in order to successfully evaluate the trained network.

4. `combined_{traffic_type}_{seed}b.json`

This is the same as above, but additionally sets TEST_ST_STRICTLY_BETTER=True, which enables the part of our algorithm that compares the efficiency of the ST solver's path and the DDPG's projected path.

5. `ddpg_{traffic_1}_network_{traffic_2}_traffic_{seed}.json`

This evaluates the DDPG trained on `traffic_1` using the traffic type specified in `traffic_2`. You need to run `train_{traffic_1}_{seed}` in order to successfully evaluate this combination.

6. `cross_{traffic_1}_network_{traffic_2}_traffic_{seed}(b).json`

This evaluates the combination of the ST solver and the DDPG trained on `traffic_1`, using the type specified in `traffic_2`.  You need to run `train_{traffic_1}_{seed}` in order to successfully evaluate this combination. If a final `b` is present, then we also set TEST_ST_STRICTLY_BETTER=True (to see the effect of that part of the algorithm).