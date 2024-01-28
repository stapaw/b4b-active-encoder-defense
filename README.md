# Bucks for Buckets (B4B): Active Defenses Against Stealing Encoders

<img width="1362" alt="image" src="https://github.com/stapaw/active-ssl/assets/18675688/4d7ecfd8-8c39-4234-9824-bcbf6bd92d9b">

Machine Learning as a Service (MLaaS) APIs provide ready-to-use and high-utility encoders that generate vector representations for given inputs. Since these encoders are very costly to train, they become lucrative targets for model stealing attacks during which an adversary leverages query access to the API to replicate the encoder locally at a fraction of the original training costs. We propose Bucks for Buckets (B4B), the first active defense that prevents stealing while the attack is happening without degrading representation quality for legitimate API users. Our defense relies on the observation that the representations returned to adversaries who try to steal the encoder's functionality cover a significantly larger fraction of the embedding space than representations of legitimate users who utilize the encoder to solve a particular downstream task. B4B leverages this to adaptively adjust the utility of the returned representations according to a user's coverage of the embedding space. To prevent adaptive adversaries from eluding our defense by simply creating multiple user accounts (sybils), B4B also individually transforms each user's representations. This prevents the adversary from directly aggregating representations over multiple accounts to create their stolen encoder copy. Our active defense opens a new path towards securely sharing and democratizing encoders over public APIs.

### Setup
* Create environment with python 3.9
* Install requirements from requirements.txt
* Install and [setup Wandb](https://docs.wandb.ai/quickstart) (pipreqs does not add it to requirements)
```bash
pip install wandb
```
* Make sure you have access to project given by `<wandb_entity>/<wandb_project>` from [config](config)
* Set [src](src) directory as PYTHONPATH
```bash
export PYTHONPATH=${PYTHONPATH}:./src
```
* Download SimSiam ResNet50 checkpoint pretrained on ImageNet from https://dl.fbaipublicfiles.com/simsiam/models/100ep/pretrain/checkpoint_0099.pth.tar and place in ./end2end_stealing/pretrained_weights
* Download DINO VitS16 pretrained on ImageNet from https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth and place in ./end2end_stealing/pretrained_weights
* Download a subset of ImagNet1k with 100 examples per class in the train set from https://drive.google.com/file/d/1bJ_LmzUR-CsrWpVORoWQ0NWU46EpBhWT/view?usp=sharing

## Running end2end experiments

### Run model stealing
Stealing an undefended SimSiam ResNet50 encoder
```bash
./bash_scripts/steal_undefended_simsiam.sh
```
Stealing a SimSiam ResNet50 encoder defended with B4B
```bash
./bash_scripts/steal_defended_simsiam.sh
```
Stealing a SimSiam ResNet50 encoder defended with B4B using 2 sybil accounts
```bash
./bash_scripts/steal_defended_simsiam_sybil.sh
```

### Run evaluation of stolen models on downstream tasks
```bash
./bash_scripts/evaluate_stolen_undefended_simsiam.sh
./bash_scripts/evaluate_stolen_defended_simsiam.sh
./bash_scripts/evaluate_stolen_defended_simsiam_sybil.sh
```

## Running experiments on transformations
**A.** Before you run experiments you should use scripts in [src/data/embeddings](src/data/embeddings)
to save train and test embeddings from SimSiam or DINO in .pt format 
(paths to those files are input parameters for sybil_defense.py script and linear classifier evaluations.)

**B.** Main script to run experiment evaluating mapping representations from different accounts is [src/sybil_defense.py](src/sybil_defense.py).
Running a script results in saving in the output file dictionary with accuracy and cosine similarity obtained after the mapping of representations for each subset of N queries. List of different number of queries is controlled with subsets argument.

The setup for the remapping is presented below (Figure 13):  


<img width="1237" alt="remapping_setup" src="https://github.com/stapaw/active-ssl/assets/18675688/234d4979-61d5-44b4-98fd-c70771251939">

The protocol of evaluating remappings for two sybil accounts:
 
1. API receives inputs from two sybil accounts and generates corresponding representations.
 
2. Representations are transformed on a per-user basis and returned.
 
3. Adversary trains a reference classifier on representations from account one.
 
4. Adversary trains a linear model to find mapping from representations of account two to representations of account one.
 
5. To check the quality of obtained mapping representations from test set of account two are mapped using the fixed mapper (from step 4) to representation space of account one. This enables the calculation of cosine distance between representations from account one and their counterparts from account two (Figure 5 in publication). Additionally, the fixed reference classifier (from step 3) can be used to measure the accuracy drop caused by remapping . 

**C.** Scripts named reference_* are used for training linear classifier to evaluate the usefullness of representations on downstream tasks (with or without additional transformations). They also need paths to embeddings (for different datasets, noised or clean) obtained in A.

**D.** We use hydra to manage experiments. Experiment parameters are stored in [config](config) directory. 

* To run training with default config, run the command below. 
Experiment parameters will be loaded from [config/config.yaml](config/config.yaml).
```shell
python src/sybil_defense.py
```
* To override experiment parameters, use dot notation.
```shell
python src/sybil_defense.py \
  seed=0 \
  benchmark.hparams.train_epochs_reference_classifier=10 \
  benchmark.hparams.lr_mapper = 0.0001
```
* To use load partial configs for benchmarks etc. use the notation from below.
Here *benchmark* is the name of the [directory](config/benchmark) in [config](config) and 
*mnist_embeddings* is the name of file in this directory (without .yaml extension).
```shell
python src/sybil_defense.py \
  benchmark=benchmark.mnist_embeddings
```

[comment]: <> (wandb sweep sweeps/experiment.yaml)

[comment]: <> (```)

```
