import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

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

    def prepare_data(self, data):
        """
        Prepare the data by splitting it into training and testing sets.
        """
        self.X = data.drop(columns=[self.target_column], axis=1)
        self.y = data[self.target_column]


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

    def train_model(self, print_summary=True):
        """
        Train the logistic regression model using the training data.
        """
        self.model = sm.Logit(self.y_train, self.X_train)
        self.result = self.model.fit(method='bfgs', maxiter=1000)

        if print_summary:
            print(self.result.summary())

    def evaluate_model(self, selected_threshold=0.5):
        """
        Evaluate the model's performance on the test data.
        """
        predictions = self.result.predict(self.X_test)
        predictions = (predictions >= selected_threshold).astype(int)
        accuracy = accuracy_score(self.y_test, predictions)
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, predictions))
        print("Classification Report:")
        print(classification_report(self.y_test, predictions))

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