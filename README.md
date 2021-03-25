# Property-oriented Design Framework for Energetic Materials
Property-oriented Design Framework for Energetic Materials (PDFEM) can quickly design new energetic compounds with desired property based on limited-scale dataset labeled.

## Prerequisites
This package requires:
* Python
* Jupyter Notebook
* Scikit-learn

## Usage
### Construct a search sapce
The implementation of PDFEM begins with the construction of the search space by means of `Code/get_searchspace.ipynb`, `Data/parent rings.xlsx` and `Data/substituents.xlsx`.

### Train a machine learning model
The package provides the training process of SVR.lin as an example in `Code/SVR.lin.ipynb`. The descriptors used in PDFEM are <br>organized in `Code/descriptors`.

### Construct an optimizer
Before utilizing the optimizer to recommend candidates from the unexplored search space, it's necessary to investigate the <br>searching efficiency of the optimizer 
on the initial dataset, as presented in `Code/Trade-off`.

### Iteration
Based on the combination of the machine learning model and the optimizer, PDFEM starts to iteratively recommend candidate samples from the search space. 
`Code/iteration_1` represents the first iteration of PDFEM utilizing SVR.lin/Trade-off coupled with SOB+Estate descriptors.
