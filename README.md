# Binary Classification Comparison

## Overview
This project compares three supervised learning methods - KNN, Logistic Regression, and Decision Tree - on a binary classification problem using Python.

## Introduction
Supervised learning involves training algorithms on labeled data. This project aims to compare the performance of these algorithms.

## Justification of Libraries Used
- **Pandas:** Data manipulation and loading datasets.
- **Numpy & Matplotlib:** Mathematical operations and data visualization.
- **Seaborn:** High-level visualization for correlation matrices.
- **Scikit-learn:** Crucial for data splitting, cross-validation, importing classifiers, evaluation metrics, and confusion matrices.

## Training and Testing Process
- Visualized data with histograms and correlation matrices.
- Split data into training and testing sets (80:20 ratio).
- Used scikit-learn to train each model and evaluate their performance on the testing set.
- Implemented K-Fold Cross Validation (k=5) to tune hyperparameters and prevent overfitting.

## Tuning the Hyperparameters
- Cross-validated the training data to find optimal hyperparameters for each algorithm.
- Plotted cross-validation results against hyperparameters to select the best values.

![KNN](https://github.com/sacadelmi/Supervised-Learning/blob/main/knn_evaluation.png)
![Logistic Regression](https://github.com/sacadelmi/Supervised-Learning/blob/main/logisticr_evaluation.png)
![Decision Tree](https://github.com/sacadelmi/Supervised-Learning/blob/main/decisiontree_evaluation.png)

## Evaluation
### Confusion Matrix
- Evaluated algorithm performance using confusion matrices on unseen test data.
- Detailed breakdown of TP, FP, TN, FN for each model.

![KNN Confusion Matrix](https://github.com/sacadelmi/Supervised-Learning/blob/main/knn_confusion_matrix.png)
![Logistic Regression Confusion Matrix](https://github.com/sacadelmi/Supervised-Learning/blob/main/logisticr_confusion_matrix.png)
![Decision Tree Confusion Matrix](https://github.com/sacadelmi/Supervised-Learning/blob/main/decisiontree_confusion_matrix.png)

### Evaluation Metrics
- Calculated precision, recall, F1-score, and accuracy for each model on the testing data.

## Evaluation Metrics (Test Data)
| Metric            | KNN   | Logistic Regression | Decision Tree |
|-------------------|-------|---------------------|---------------|
| Precision         | 0.977 | 0.977               | 0.830         |
| Recall            | 0.957 | 0.913               | 0.957         |
| F1-Score          | 0.967 | 0.944               | 0.889         |
| Accuracy          | 0.979 | 0.964               | 0.921         |

## Conclusion
### Analysis of Models
Based on my analysis of the confusion matrix and evaluation metrics I have deduced that all three of my algorithms performed well on the unseen training dataset. The common advantage of each of my algorithms is that they were all simple to understand and implement. KNN had the highest performance on all evaluation metrics. The model is effective in identifying non-linear decision boundaries between classes. However, the cons of the model are that it can be sensitive to the choice of k and it can be computationally expensive. 

Logistic Regression also performed nearly as well as KNN, however cons of the algorithm suggest that it can be quite sensitive to outliers. Decision Tree underperformed comparatively to my other two algorithms this could be due to overfitting. Also, to note a common con of the Decision Tree is that it is sensitive to small variations in the training data.

### Personal Reflection
- Experience gained with scikit-learn and understanding cross-validation.
- Desire to explore more classifiers in the future and work with multiclass data.

This project served as an excellent introduction to supervised learning and the evaluation of classification models.

