# Credit Risk Analysis

## Project Overview
The purpose of this project is to use different techniques to train and evaluate models with unbalanced classes .Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then,  use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next,  compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk.Several different algorithms are used to predict credit risk. The performance of these different models are compared and recommendations are suggested based on the results.

## Software
- Python 3.7
- SciPy 1.6.2
- Scikit-learn 0.1
- imbalanced-learn 0.80


## Results

In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced.

- Balanced Accurracy Score: 0.674
- High-Risk Precision: 0.01
- High-Risk Recall: 0.74


### Oversampling: Native Random Oversampling
Synthetic Minority Oversampling Technique (SMOTE) is an oversampling approach to deal with unbalanced datasets.

- Balanced Accurracy Score: 0.662
- High-Risk Precision: 0.01
High-Risk Recall: 0.63



### Oversampling: SMOTE
In random undersampling, instances are randomly selected from the majority class and removed until the size of the majority class is reduced (typically to the size of the minority class).

Balanced Accurracy Score: 0.662
High-Risk Precision: 0.01
High-Risk Recall: 0.63

### Undersampling: Random Undersampling

In random undersampling, instances are randomly selected from the majority class and removed until the size of the majority class is reduced (typically to the size of the minority class).

Balanced Accurracy Score: 0.662
High-Risk Precision: 0.01
High-Risk Recall: 0.63

### Combination Sampling: SMOTEENN
In random undersampling, instances are randomly selected from the majority class and removed until the size of the majority class is reduced (typically to the size of the minority class).

Balanced Accurracy Score: 0.662
High-Risk Precision: 0.01
High-Risk Recall: 0.63

### Balanced Random Forest Classifier

A Balanced Random Forest is an ensemble method that randomly under-samples to achieve balance.

Balanced Accurracy Score: 0.788
High-Risk Precision: 0.03
High-Risk Recall: 0.70

### Easy Ensemble AdaBoost Classifier
Easy Ensemble AdaBoost Classifier
Bag of balanced boosted learners also known as EasyEnsemble. The balancing is achieved by random under-sampling.

Balanced Accurracy Score: 0.931
High-Risk Precision: 0.09
High-Risk Recall: 0.92


## Summary

For all models, Easy Ensemble AdaBoost Classifier has the highest Balanced Accuracy Score out of all of the techniques employed in this project.

it is recommended to use Easy Ensemble AdaBoost Classifier. Although the EasyEnsemble model performed the best in this group, its precision for high-risk data points is still only 0.09. This indicates that very few of positive predictions are true (False Positives). 
