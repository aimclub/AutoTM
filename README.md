
<img align="left" src="docs/img/MyLogo.png" alt="Library scheme" height="180"/>

# AutoTM

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
![build](https://github.com/ngc436/AutoTM/actions/workflows/build.yaml/badge.svg)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PyPI version](https://badge.fury.io/py/autotm.svg)](https://badge.fury.io/py/autotm)

Automatic parameters selection for topic models (ARTM approach) using evolutionary algorithms. 
AutoTM provides necessary tools to preprocess english and russian text datasets and tune topic models.

## What is AutoTM?
Topic modeling is one of the basic methods for EDA of unlabelled text data. While ARTM (additive regularization 
for topic models) approach provides the significant flexibility and quality comparative or better that neural 
approaches it is hard to tune such models due to amount of hyperparameters and their combinations.

To overcome the tuning problems AutoTM presents an easy way to represent a learning strategy to train specific models for input corporas.

<img src="docs/img/strategy.png" alt="Learning strategy representation" height=""/>

Optimization procedure is done by genetic algorithm which operators are specifically tuned for 
the task. To speed up the procedure we also implemented surrogate modeling that, for some iterations, 
approximate fitness function to reduce computation costs on training topic models.

<img src="docs/img/img_library_eng.png" alt="Library scheme" height=""/>


## Installation

! Note: The functionality of topic models training is available only for linux distributions.

Via pip:

```pip install autotm```

From source:

```pip install -r requirements.txt```  

```python -m spacy download en_core_web_sm```

```export PYTHONPATH="${PYTHONPATH}:/path/to/src"```

[//]: # (## Dataset and )

## Quickstart

The notebook with an example is available in ```examples``` folder.

## Running from the command line

To fit a model:
```autotmctl --verbose fit --config conf/config.yaml --in data/sample_corpora/sample_dataset_lenta.csv```

To predict with a fitted model:
```autotmctl predict --in data/sample_corpora/sample_dataset_lenta.csv --model model.artm```

## Distributed version

Distributed version to run experiments on kubernetes is available in ```autotm-distributed``` brunch. Still this version is in development stage and will be transfered to separate repository.

## Backlog:
- [ ] Add tests
- [ ] Add new multi-stage 
 
## Citation

```bibtex
@article{10.1093/jigpal/jzac019,
    author = {Khodorchenko, Maria and Butakov, Nikolay and Sokhin, Timur and Teryoshkin, Sergey},
    title = "{ Surrogate-based optimization of learning strategies for additively regularized topic models}",
    journal = {Logic Journal of the IGPL},
    year = {2022},
    month = {02},
    issn = {1367-0751},
    doi = {10.1093/jigpal/jzac019},
    url = {https://doi.org/10.1093/jigpal/jzac019},}

```
