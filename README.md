# Property-oriented Adaptive Design Framework
Property-oriented Adaptive Design Framework (PADF) can quickly design new energetic compounds with desired property based on limited-scale dataset labeled.

## Prerequisites
This package requires:
* Python
* Jupyter Notebook
* Scikit-learn

## Usage
### Construct a search sapce
The implementation of PADF begins with the construction of the search space by means of `Code/get_searchspace.ipynb`.

### Train a machine learning model
The training process of SVR.lin is shown as an example in `Code/SVR.lin.ipynb`. The descriptors used in PADF are organized in `Code/descriptors.py`.
You can name your data as "initial dataset.xlsx" and use this dataset to train the machine learning model.

### Construct an optimizer
Before utilizing the optimizer to recommend candidates from the unexplored search space, it's necessary to investigate the searching efficiency of the optimizer on the initial dataset, as presented in `Code/trade-off.ipynb`.

### Iteration
Based on the combination of the machine learning model and the optimizer, PADF starts to iteratively recommend candidate samples from the search space. `Code/iteration_1.ipynb` represents the first iteration of PADF utilizing SVR.lin/Trade-off coupled with SOB+Estate descriptors.
