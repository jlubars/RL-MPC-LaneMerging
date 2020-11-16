# Combining  Reinforcement  Learning  with  Model  Predictive  Control  for On-Ramp  Merging

This repository contains the code used for the experiments in our paper of the above name, along with some experimental results (soon!)

### Setup instructions:
1. Install SUMO version 1.7.0 and set SUMO_HOME to point to that installation
2. Install python 3.6 or above
3. Clone this repository
4. (If desired) Create a virtual environment for this project and activate it
    ```
    python3 -m venv ./mergeEnv/
    source mergeEnv/bin/activate
    ```
5. (If CUDA support is desired) Install PyTorch with CUDA support, following the instructions on the [PyTorch website](https://pytorch.org/).
6. Install the other requirements with pip
    ```
    cd MergeControl
    pip install -r requirements.txt
    ```
7. Compile the Cython:
    ```python setup.py build_ext --inplace```

### Running experiments:
Run `main.py`, using the name of the desired configuration file as an argument. All configuration files used for our experiments can be found in the configs folder.
```
python main.py configs/<settings>.json
```
The results of each experiment will be appended to a file named `run_data.csv`.

### Full experimental results from the paper:
See the `experiment_data` folder
