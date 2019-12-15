<!-- <p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://i.imgur.com/6wj0hh6.jpg" alt="Project logo"></a>
</p> -->

<h1 align="center">LDR</h1>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/ekrekr/ldr.svg)](https://github.com/ekrekr/ldr/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/ekrekr/ldr.svg)](https://github.com/ekrekr/ldr/pulls)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

</div>

---

LDR stands for **Latent Dimensionality Reduction**. It is a generic method for interpreting models. It is deployed here as a python module.

## About

The purpose of LDR is to solve a common and [controversial](https://arxiv.org/abs/1811.10154) problem. Often models that have a higher predictive accuracy are more complex. These complex models are sensibly referred to as **black box models**. This is frustrating for many data scientists, as they end up with a model that performs well, but they **can't explain why**, which can cause the model to fail in critical situations which are difficult to test for.

LDR aims to bridge that gap by providing a generic, reliable algorithmic method for interpreting most models. I define interpretability as:

1. Understanding the quality of a systems current understanding. Is the model overfitting, is there insufficient training data, and thus will it fail when deployed to the real world?

2. Interpreting how the value of a feature, or subset of features, affects a model's prediction (which I refer to here as **feature interpretation**).

3. The ability to use a model when not all values for the input features are present.

## Getting Started

### Prerequesites

[Python3](https://www.python.org/download/releases/3.0/).

### Installation

```console
python3 -m pip install ldr --user
```

### Usage

An example analysis of a simple generated distribution can be found [here](distribution_example.ipynb).

An example analysis of a classification problem can be found [here](classification_example.ipynb).

An example analysis of a regression problem can be found [here](regression_example.ipynb).

A step by step multi-model interpolation example can be found, where outlier detection is enforced to improve the efficacy for critical systems [here](interpolation_example.ipynb).

## Additional Notes

If you find this package useful, please consider [contributing](contributing.md)!
