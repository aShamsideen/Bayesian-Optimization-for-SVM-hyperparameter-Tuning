# Bayesian-Optimization-for-SVM-hyperparameter-Tuning

This project demonstrates how to use Bayesian Optimization to tune hyperparameters of a Support Vector Machine (SVM) classifier on the Breast Cancer dataset from scikit-learn.

The optimization process aims to maximize the ROC AUC score, ensuring the model achieves strong discriminative performance between malignant and benign cases.


📌 Features of the Code

* Loads the Breast Cancer dataset from sklearn.datasets.

* Preprocesses features using MinMaxScaler for normalization.

* Defines a black-box function that evaluates SVM performance for given hyperparameters.

* Uses Bayesian Optimization (bayes_opt) to efficiently search the hyperparameter space.

* Supports single-parameter optimization (e.g., C) and multi-parameter optimization (e.g., C and degree).

* Implements Upper Confidence Bound (UCB) acquisition strategy for balancing exploration and exploitation.

* Visualizes optimization progress across iterations.

  

🚀 Dependencies

Make sure you have the following Python packages installed:
pip install numpy matplotlib scikit-learn bayesian-optimization



⚙️ Code Workflow
1. Data Preparation

  * Load the dataset

  * Split into train/test sets with stratification

  * Apply MinMax scaling

2. Black-Box Function

The function trains an SVC with given hyperparameters (C, degree, etc.), evaluates it on the test set, and returns the ROC AUC score.

def black_box_function(C, degree):
    model = SVC(C=C, degree=degree)
    model.fit(X_train_scaled, y_train)
    y_score = model.decision_function(X_test_scaled)
    return roc_auc_score(y_test, y_score)

3. Bayesian Optimization

Define parameter search space:
pbounds = {"C": [0.01, 10], "degree": [1, 5]}


* Use the UCB acquisition function to guide exploration:
  utility = UtilityFunction(kind="ucb", kappa=1.96, xi=0.01)


* Run iterative optimization:
  for i in range(25):
    next_point = optimizer.suggest(utility)
    next_point["degree"] = int(next_point["degree"])
    target = black_box_function(**next_point)
    optimizer.register(params=next_point, target=target)


4. Results & Visualization

* Best hyperparameters and corresponding ROC AUC are printed:
  print("Best result: {}; f(x) = {:.3f}".format(optimizer.max["params"], optimizer.max["target"]))

* Plot optimization progress:


📊 Example Output:
Best result: {'C': 4.23, 'degree': 3}; f(x) = 0.994

This means the optimizer found that an SVM with C=4.23 and degree=3 achieved an ROC AUC score of 0.994.


🔮 Extensions

* Add more hyperparameters to optimize (e.g., kernel, gamma).

* Use Expected Improvement (EI) or Probability of Improvement (PI) as acquisition functions.

* Run cross-validation instead of a single train-test split for more robust evaluation.
