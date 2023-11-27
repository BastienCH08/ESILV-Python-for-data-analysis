Python For Data-Analysis

Overview

This project involves the analysis of obesity-related data, including preprocessing, visualization, and modeling. The analysis is conducted in a Jupyter notebook (project.ipynb), followed by the development of a Flask API to showcase the project.

The team members are:

CHEREL Bastien
CHOUAKI Amine
THEAGENE Yann

0. Contents

1. Analysis
- Preprocessing
- Visualization
- Modeling
2. Flask API
- Usage


1. Analysis

Preprocessing

The dataset is loaded from a CSV file (ObesityDataSet.csv) and undergoes preprocessing steps such as column renaming, rounding values, and the calculation of Mass Body Index (MBI). Categorical variables are mapped to meaningful labels.

Visualization

The analysis includes visualizations using Matplotlib and Seaborn. Subplots are created for obesity distribution, age vs. MBI by obesity, gender vs. obesity, and obesity types by physical activity.

Modeling

A Jupyter notebook (project.ipynb) includes the implementation of machine learning models using Random Forest, Gradient Boosting, and Naive Bayes. Hyperparameter tuning is performed using GridSearchCV, and the models are evaluated and compared based on accuracy, R-squared, and confusion matrix.

2. Flask API

Through the API, users can access visualizations, explore the results of machine learning models, and gain a deeper understanding of the patterns observed in the dataset. 