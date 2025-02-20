# ComparativeAnalysis_of_ClassificationAlgorithms

## Objective
This project aims to implement and compare various machine learning classifiers for predictive tasks, specifically focusing on K-Nearest Neighbors (KNN) and Naive Bayes algorithms. The study evaluates their performance using 10-fold cross-validation on two datasets: the Pima Indian Diabetes Dataset (health-related predictions) and the Room Occupancy Dataset (sensor-based occupancy detection). Additionally, Weka, a machine learning tool, is used to facilitate comparative analysis among multiple classifiers, offering insights into their strengths and weaknesses across different datasets.

## Methods and Tools Used
The study employed Python and Weka for machine learning model implementation, evaluation, and comparison. Data preprocessing, normalization, and feature selection were performed to optimize model accuracy. The following classifiers were analyzed:

- Baseline Classifier (ZeroR): Establishes a simple majority-class baseline.
- Decision Trees (DT): Provides interpretable classification models.
- Naive Bayes (NB): Assesses probabilistic relationships among variables.
- K-Nearest Neighbors (1NN & 5NN): Measures classification accuracy based on nearest neighbors.
- Support Vector Machines (SVM): Identifies linear separability between classes.
- Multi-layer Perceptron (MLP): Evaluates complex, non-linear relationships.
- Random Forest (RF): Analyzes ensemble learning effectiveness.
Each model was tested on both datasets using 10-fold cross-validation, and their accuracy was compared against Weka's default implementations.

## Findings and Conclusion
The results showed dataset-dependent classifier performance. For the Pima Indian Diabetes Dataset, SVM achieved the highest accuracy (76.30%), followed by Naive Bayes (75.13%) and MLP (75.39%), indicating that complex models perform better for medical data with non-linear relationships. In contrast, for the Room Occupancy Dataset, Random Forest (99.65%) and Decision Trees (high accuracy, fast training) outperformed others, benefiting from well-separated sensor data. The study concluded that choosing the right classifier depends on dataset complexity and structure, with simpler models performing well on structured data and more sophisticated models excelling in complex, interdependent datasets. Future work could explore deep learning models and additional datasets to refine classification strategies further.
