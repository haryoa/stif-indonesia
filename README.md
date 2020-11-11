# STIF-Indonesia

Implementation of "Semi-Supervised Low-Resource Style Transfer of Indonesian Informal to Formal Language with Iterative Forward-Translation".

We change the data where it is different than the data published in the paper. We expect you to find a different result.

To be denounced, please wait!

## Paper

Semi-Supervised Low-Resource Style Transfer of Indonesian Informal to Formal Language with Iterative Forward-Translation (IALP 2020)

## Requirements

we use the Ubuntu 17.04+ Moses which only works on the specified OS.

If you use other moses, please change the `scripts/download_moses.sh`

```bash
curl http://www.statmt.org/moses/RELEASE-4.0/binaries/ubuntu-17.04.tgz -o moses.tgz
```

to

```bash
curl [OTHER MOSES URL] -o moses.tgz
```

In this experiment, we wrap the MOSES code by using Python's `subprocess`. So a python installation is necessary. The system is tested on Python 3.9. We recommend it to install with `miniconda`. You can install it by following this link: https://docs.conda.io/en/latest/miniconda.html

## How To Run

First, clone the repository

```bash
git clone https://github.com/haryoa/stif-indonesia.git
```

Then run the MOSES downloader. We use .sh, so use a CLI applications that can execute it. On the root project folder directory, do:

```bash
bash scripts/download_moses.sh
```

The script will download the moses toolkit and extract it by itself.

### Run Supervised Experiments

To run the supervised one, do:

```bash
python -m stif_indonesia --exp-scenario supervised
```

It will read the experiment config in `experiment-config/00001_default_supervised_config.json`

### Run Semi-Supervised Experiments

To run the semi-supervised one, do:

```bash
python -m stif_indonesia --exp-scenario semi-supervised
```

It will read the experiment config in `experiment-config/00002_default_semi_supervised_config.json`

## Output

1. The training process will output the log of the experiment in `log.log`
2. The output of the model will be produced in `output` folder

### Supervised output

It will output `evaluation`, `lm` , and `train`. `evaluation` is the result of  prediction on the test set, `lm` is the output of the trained LM, and `train` is the produced model by the moses toolkit

### Semi supervised output

It will output `agg_data`, `best_model_dir`, and `produced_tgt_data`. `agg_data` is the result of the forward-iteration data synthesis. `best_model_dir` is the best model produced by the training process, and `produced_tgt_data` is the prediction output of the test set.

### Score

Please check the `log.log` file which is the output of the process. 

## TODO Write

1. Link to arxiv + short description
2. Acknowledgement
3. Team

