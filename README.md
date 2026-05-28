# Spam Classification with Machine Learning

This repository contains a complete SMS spam classification assignment implemented in Python and documented in a Jupyter notebook. The project focuses on detecting whether a text message is legitimate (`ham`) or unsolicited (`spam`) using classical machine learning, with the full modeling pipeline explained step by step.

## Project Overview

The notebook covers the full workflow for binary text classification:

- loading and inspecting an SMS dataset,
- preprocessing raw text,
- converting messages into numeric features,
- implementing a Decision Tree classifier from scratch,
- evaluating model performance on a held-out test set,
- and improving the model through grid search with 5-fold cross-validation.

The work is designed to be both practical and educational. In addition to training a usable classifier, the notebook explains the underlying mathematics and implementation details behind preprocessing, TF-IDF, impurity-based splitting, and F1-based model selection.

## Dataset

The dataset used in this project is included in the repository:

- `SPAM text message 20170820 - Data.csv`

Each record contains:

- `Category`: `ham` or `spam`
- `Message`: raw SMS text

The notebook uses a fixed split:

- Training set: 4,460 messages
- Test set: 1,112 messages

This dataset is imbalanced, with ham messages forming the majority class. Because of that, the notebook prioritizes F1 score for spam detection rather than relying on accuracy alone.

## Methods Used

### Text preprocessing

The preprocessing pipeline includes:

- lowercasing,
- punctuation removal,
- tokenization,
- and stopword removal.

### Feature extraction

Two feature engineering approaches are implemented from scratch:

- Bag of Words (BoW)
- TF-IDF

Both feature spaces are binarized before training so the Decision Tree can split meaningfully on token presence versus absence.

### Model

The classifier is a custom binary `DecisionTreeClassifier` implemented with `numpy`. It supports:

- Gini impurity,
- entropy,
- configurable tree depth,
- and configurable minimum samples required for splitting.

### Model selection

The notebook runs a manual grid search with 5-fold cross-validation over:

- feature method: `bow` or `tfidf`
- tree depth: `3`, `5`, `10`
- split criterion: `gini` or `entropy`
- minimum samples split: `2` or `5`

## Best Model

The best-performing configuration found by grid search was:

- Feature method: `TF-IDF`
- Max depth: `10`
- Criterion: `gini`
- Minimum samples split: `2`
- Average CV F1: `0.7409`

## Final Test Performance

Using the best configuration above, the final model achieved:

- Precision: `0.8532`
- Recall: `0.6414`
- F1 Score: `0.7323`
- Accuracy: `0.9388`

Confusion counts on the held-out test set:

- True Positives: `93`
- True Negatives: `951`
- False Positives: `16`
- False Negatives: `52`

These results show strong overall accuracy and good precision, with room for improvement in recall on spam messages.

## Repository Contents

- `spam_classification_assignment.ipynb`: main notebook containing the full assignment, implementation, experiments, and results
- `SPAM text message 20170820 - Data.csv`: dataset used by the notebook
- `README.md`: project summary and usage notes

## How to Run

1. Open `spam_classification_assignment.ipynb` in Jupyter Notebook or JupyterLab.
2. Ensure Python has the required libraries installed:
   - `numpy`
   - `pandas`
   - `matplotlib`
   - `scikit-learn`
3. Confirm that the dataset path in the notebook matches your local environment if needed.
4. Run the notebook cells in order.

## Notes

- The notebook includes explanatory markdown intended for academic review as well as implementation detail.
- The final demo cell uses the retrained final model objects (`final_vectorizer` and `final_clf`) and includes a guard for messages that produce no known vocabulary matches.

## Author

Jabiev
