# E-Commerce Transaction Fraud Detection Using Machine Learning

## Overview

This project focuses on detecting fraudulent e-commerce transactions using Machine Learning techniques. The dataset is preprocessed, analyzed through Exploratory Data Analysis (EDA), and multiple classification algorithms are evaluated to identify fraudulent transactions.

## Features

* Data preprocessing and cleaning
* Exploratory Data Analysis (EDA)
* Transaction type analysis
* Fraud vs Non-Fraud visualization
* Feature scaling and encoding
* Model comparison using:

  * Logistic Regression
  * Random Forest
  * XGBoost
* Threshold optimization
* Fraud risk classification system
* Model serialization using Joblib

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Joblib

## Project Structure

Data/ – Dataset files

models/ – Saved machine learning models and preprocessing objects

results/ – Graphs, visualizations, and confusion matrices

analysis.py – Exploratory Data Analysis and visualizations

train_model.py – Model training, evaluation, and saving

## Exploratory Data Analysis

The project includes:

* Transaction type distribution
* Fraud vs Non-Fraud distribution
* Fraud transaction rate by type
* Transaction amount analysis
* Fraud trend analysis over time
* Correlation heatmap

## Models Evaluated

1. Logistic Regression
2. Random Forest
3. XGBoost

XGBoost was selected as the final model due to its superior performance on fraud detection.

## Final XGBoost Performance

* Precision: 85.5%
* Recall: 82.1%
* F1 Score: 83.8%

## Fraud Risk Tiers

The system categorizes transactions into:

* Approve
* Soft Flag
* Manual Review
* Block

This tier-based approach helps reduce false positives while maintaining fraud detection effectiveness.

## Author

Jatin Mishra

