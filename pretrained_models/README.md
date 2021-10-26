These are some pretrained models, trained using the `train_{traffic_type}_{seed}.json` configs. To use them for the configs that require a trained DDPG model, copy the appropriate model to `runs`. For example:

```shell
cp pretrained_models/ddpg_default1_extended runs -r
python main.py configs/combined_default_1.json
```

Each of these models should be able to be recreated using the appropriate `train_{traffic_type}_{seed}.json` config file, though differences in hardware and versions of pytorch, sumo, and numpy may affect the RNG used for training.