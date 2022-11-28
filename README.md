# AutoTM

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
![build](https://github.com/ngc436/AutoTM/actions/workflows/build.yaml/badge.svg)

Automatic parameters selection for topic models (ARTM approach) using evolutionary algorithms. 
AutoTM provides necessary tools to preprocess english and russian text datasets and tune topic models.

## Installation

! Note: The functionality of topic models training is available only for linux distributions.

```pip install -r requirements.txt```

## Quickstart

The notebook with an example is available in ```examples``` folder.

## Distributed version

Distributed version to run experiments on kubernetes is available in ```autotm-distributed``` brunch. Still this version is in development stage and will be transfered to separate repository.

## Citation

```bibtex
@article{10.1093/jigpal/jzac019,
    author = {Khodorchenko, Maria and Butakov, Nikolay and Sokhin, Timur and Teryoshkin, Sergey},
    title = "{ Surrogate-based optimization of learning strategies for additively regularized topic models}",
    journal = {Logic Journal of the IGPL},
    year = {2022},2
    month = {02},
    issn = {1367-0751},
    doi = {10.1093/jigpal/jzac019},
    url = {https://doi.org/10.1093/jigpal/jzac019},}

```