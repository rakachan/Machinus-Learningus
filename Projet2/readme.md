# Machine Learning - Project 2

This is the repository for the Machine Learning EPFL course second project.

This file explains the organisation and functions of the python scripts. For more information about the implementation, see the [PDF reports] and the commented code.

First, you should place `data_train.csv` and `sample_submission.csv` at the root folder of the project.

You will also need to install PySpark and FindSpark

### Project 2.ipynb

Python notebook used to train the model of this project.
This notebook keeps track of all the methods and models we tried for this project: 
SGD, ALS, ALS with bias, Nimfa library and PySpark library

### plots.py

Contain 2 functions to plot useful things:
- **`plot_raw_data`**: plot the statistics result on raw rating data.
- **`plot_train_test_data`**: visualize the train and test data.

### helpers.py
Contain multiple methods for data processing and utilitary methods necessary to achieve the factorization:
- **`read_txt`** Reads a file as raw text 
- **`load_data`** Load data in text format, one rating per line, as in the kaggle competition.
- **`preprocess_data`** Preprocessing the text data, conversion to numerical array format.
- **`group_by`** Group list of list by a specific index.
- **`build_index_groups`** Build groups for non-zero rows and cols.
- **`calculate_mse`** Calculates mean square error.
- **`row_col_spark`** Used to interpret strings given a regular expression (used for strings of the form "r34_c4" -> (33, 3))
- **`create_csv_submission`** Generates the submission csv

### run.py
Script that generates the exact CSV file submitted on Kaggle, usingt PySpark.
