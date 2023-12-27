Binary Classification Comparison

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

## Evaluation
### Confusion Matrix
- Evaluated algorithm performance using confusion matrices on unseen test data.
- Detailed breakdown of TP, FP, TN, FN for each model.

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
- All algorithms performed well, but KNN showed the highest performance across evaluation metrics.
- Pros and cons highlighted for each algorithm based on their performance and characteristics.

### Personal Reflection
- Experience gained with scikit-learn and understanding cross-validation.
- Desire to explore more classifiers in the future and work with multiclass data.

This project served as an excellent introduction to supervised learning and the evaluation of classification models.

