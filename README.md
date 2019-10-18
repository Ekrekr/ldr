# LDR

LDR stands for Latent Dimensionality Reduction, and is a method for interpreting black box models. It is deployed here as a python module.

Black box models often provide better results than more interpretable methods, and brings some [quite strong opions](https://arxiv.org/abs/1811.10154). This method aims to bridge that gap by providing a generic, reliable algorithmic method for interpreting any model. I define interpretability as:

1. Understanding how a model understands the data, and whether it is similar to how a human would think of it.

2. Interpreting how the value of a feature, or subset of features, affects a model's prediction (feature interpretation).

3. The ability to use a model when not all values for the input features are present.

## Running the Code

### Prerequesites

1. [Python3](https://www.python.org/download/releases/3.0/).

2. [Jupyter Notebook](https://jupyter.org/).

### Execution

All examples are contained in notebooks, while the LDR module is [ldr.py](ldr.py). The required packages are listed in [requirements.txt](requirements.txt), and their respective distributions and licenses can be found on the [Python Package Index](https://pypi.org/). To run the code use:

1. `python3 -m pip install --requirement requirements.txt`.

2. `jupyter notebook`.

* The generated distribution example can be found [here](distribution_example.ipynb).

* The classification example can be found [here](classification_example.ipynb).

* The regression example can be found [here](regression_example.ipynb).

* The step by step interpolation example can be found [here](interpolation_example.ipynb).

## Additional Notes

The [style sheet used](style.mplstyle) is from [one of my personal repos](https://github.com/Ekrekr/ekrekr.style).
