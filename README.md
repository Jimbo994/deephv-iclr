# DeepHV Code

## Installing required packages

We provide a list of used packages in ```environment.yaml```. If you run a linux based operating system you can install the environment using:
```
conda env create -f environment.yaml
```

## Generating a dataset
Code used to generate the datasets can be found in `generate_data_geometric.py`
To generate a dataset, set parameters in `main()` function and run:
```
python generate_data_geometric.py
```
Not that this code can be lengthy to run for high dimensional cases. We also provide faster matlab code, however this requires that you have Matlab and the Matlab Python SDK installed.
We provided a (zipped) dataset for the setting of `M=5`.

## Training a model
To train a model run:
```
python run_mape_batched.py 5 128
```

This will train a model for `M=5` with `128` channels per layer. It will either load in the existing dataset if it's present, or generate the dataset.
We have provided this dataset in the `processed` directory.

## Running benchmarks
We provide a number of trained models in the `models` directory.

### Evolutionary algorithm benchmarks
To run the Evolutionary algorithm benchmarks, check the settings to select a model, problems, and dimensions and `run_pymoo_experiments.py` and run by:

```
python run_pymoo_experiments.py
```
Results are stored in `pymoo_results/`

### Bayesian optimization benchmarks

To run the Bayesian optimization benchmarks run the following files in the `botorch_code` directory, check settings to select a model, problems, dimensions, etc.

```
python qehvi.py
python qparego.py
python baseline_deephv_batched.py
```

Results are stored in `botorch_results/`
### Timing benchmark
To run timing benchmarks run:

```
python time_comparisons.py
```


