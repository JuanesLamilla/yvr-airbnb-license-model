import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

class LogitModel:
    def __init__(self, target_column, test_size=0.2, random_state=42):
        """
        Initialize the logistic regression model class.

        Parameters:
        - target_column: The name of the target column (dependent variable).
        - test_size: Proportion of the dataset to include in the test split.
        - random_state: Random seed for reproducibility.
        """
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.result = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X = None
        self.y = None
        self.X_val = None
        self.y_val = None

    def prepare_data(self, data, use_validation=False, validation_size=0.25):
        """
        Prepare the data by splitting it into training, (optional) validation, and testing sets.
        """
        self.X = data.drop(columns=[self.target_column], axis=1)
        self.y = data[self.target_column]

        # Split into train+val and test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y
        )

        if use_validation:
            # Further split train into train/val
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_temp, y_temp, test_size=validation_size, random_state=self.random_state, stratify=y_temp
            )
        else:
            self.X_train, self.y_train = X_temp, y_temp
            self.X_val, self.y_val = None, None

    def train_model(self, print_summary=True):
        """
        Train the logistic regression model using the training data.
        """
        self.model = sm.Logit(self.y_train, self.X_train)
        self.result = self.model.fit(method='bfgs', maxiter=1000)

        if print_summary:
            print(self.result.summary())

    def evaluate_model(self, selected_threshold=0.5, print_summary=True, return_types=None, eval_on="test"):
        """
        Evaluate the model's performance on the test or validation data.
        """
        if eval_on == "val":
            if self.X_val is None or self.y_val is None:
                raise ValueError("Validation set not found. Set `use_validation=True` when calling prepare_data().")
            X_eval = self.X_val
            y_eval = self.y_val
        elif eval_on == "test":
            X_eval = self.X_test
            y_eval = self.y_test
        else:
            raise ValueError("Invalid eval_on argument. Use 'val' or 'test'.")

        predictions = self.result.predict(X_eval)
        predictions = (predictions >= selected_threshold).astype(int)
        accuracy = accuracy_score(y_eval, predictions)

        if print_summary:
            print(f"Accuracy: {accuracy:.4f}")
            print("Confusion Matrix:")
            print(confusion_matrix(y_eval, predictions))
            print("Classification Report:")
            print(classification_report(y_eval, predictions))

        if return_types is None:
            return

        return_categories = {}
        if 'illegal_f1' in return_types:
            return_categories['illegal_f1'] = f1_score(y_eval, predictions, pos_label=0)
        if 'illegal_precision' in return_types:
            return_categories['illegal_precision'] = precision_score(y_eval, predictions, pos_label=0)
        if 'illegal_recall' in return_types:
            return_categories['illegal_recall'] = recall_score(y_eval, predictions, pos_label=0)
        if 'legal_f1' in return_types:
            return_categories['legal_f1'] = f1_score(y_eval, predictions, pos_label=1)
        if 'accuracy' in return_types:
            return_categories['accuracy'] = accuracy

        return return_categories


    def create_roc_curve(self):
        """
        Create and display the ROC curve for the model.
        """

        y_scores = self.result.predict(self.X_test)
        fpr, tpr, thresholds = roc_curve(self.y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    def create_precision_recall_curve(self):
        """
        Create and display the Precision-Recall curve for the model.
        """

        y_scores = self.result.predict(self.X_test)
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_scores)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', label='PR curve (area = {:.2f})'.format(pr_auc))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='upper right')
        plt.grid()
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.show()


    def find_threshold_where_fpr(self, target_fpr=0.2):
        """
        Find the threshold where the false positive rate (FPR) is approximately equal to the target FPR.

        Parameters:
        - target_fpr: The desired false positive rate.

        Returns:
        - The threshold value that achieves the target FPR.
        """
        y_scores = self.result.predict(self.X_test)
        fpr, tpr, thresholds = roc_curve(self.y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        fpr_diff = np.abs(fpr - target_fpr)
        idx = np.argmin(fpr_diff)
        selected_threshold = thresholds[idx]

        print(f"Selected threshold for FPR ≈ {target_fpr}: {selected_threshold:.4f}")
        print(f"Actual FPR: {fpr[idx]:.4f}, TPR: {tpr[idx]:.4f}")

        return selected_threshold
    
    def predict(self, new_data):
        """
        Make predictions on new data.

        Parameters:
        - new_data: DataFrame containing the new data (must match training data structure).

        Returns:
        - Predictions as a NumPy array.
        """
        return (self.result.predict(sm.add_constant(new_data)) > 0.5).astype(int)