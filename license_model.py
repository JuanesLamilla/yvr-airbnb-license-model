import pandas as pd
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from src import detect_separation, drop_column_using_vif_
from logit_model_class import LogitModel

class LicenseModel:
    def __init__(self, city, input_file, min_nights_for_license, license_regex_pattern, protected_columns=None):
        """
        Initialize the model with city-specific settings.
        """
        self.city = city
        self.input_file = input_file
        self.min_nights_for_license = min_nights_for_license
        self.license_regex_pattern = license_regex_pattern
        self.protected_columns = protected_columns or []  # None list as default, aiming for harmonize the columns across control groups and treatments
        self.listings_df = None

        self.logit_model = None


############################################################
# MARK: 1. STARTUP
############################################################


    def load_data(self, print_info=False):
        """
        Load the data from the input file.
        """
        self.listings_df = pd.read_csv(self.input_file)

    def preprocess_data(self, print_info=False):
        """
        Preprocess the data: filter, clean, and transform columns.
        """
        # Filter by minimum nights
        if 'minimum_nights' in self.listings_df.columns:
            self.listings_df = self.listings_df[self.listings_df['minimum_nights'] < self.min_nights_for_license]

        # Drop unnecessary columns
        excluded_columns = [
            'listing_url', 'scrape_id', 'last_scraped', 'source', 'id', 'name', 'description',
            'neighborhood_overview', 'picture_url', 'host_id', 'host_url', 'host_name', 'host_since',
            'host_location', 'host_about', 'host_thumbnail_url', 'host_picture_url', 'latitude', 'longitude',
            'calendar_updated', 'calendar_last_scraped', 'amenities', 'bathrooms_text', 'first_review',
            'last_review', 'neighbourhood', 'property_type', 'host_neighbourhood', 'maximum_minimum_nights',
            'maximum_nights', 'minimum_minimum_nights', 'maximum_maximum_nights', 'minimum_maximum_nights',
            'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'has_availability'
        ]
        self.listings_df = self.listings_df.drop(columns=[col for col in excluded_columns if col in self.listings_df.columns and col not in self.protected_columns]
)

        # Drop empty and constant columns
        self.listings_df = self.listings_df.dropna(axis=1, how='all')
        constant_columns = self.listings_df.columns[self.listings_df.nunique() <= 1]

        if print_info:
            print(f"Columns with constant values: {constant_columns.tolist()}")

        self.listings_df = self.listings_df.drop(columns=constant_columns)

        # If 'license' column exists, for all rows where 'license' is NaN, set it to 'unknown'
        if 'license' in self.listings_df.columns:
            self.listings_df['license'] = self.listings_df['license'].fillna('unknown')
            # Convert 'license' column to string type
            self.listings_df['license'] = self.listings_df['license'].astype(str)
        else:
            raise ValueError("The 'license' column is missing from the DataFrame. Please ensure the input file contains this column.")

    def create_legal_listing_column(self, print_info=False, overwrite_existing=False):
        """
        Create a 'legal_listing' column based on the license regex pattern.
        """
        # Create 'legal_listing' column
        if 'legal_listing' in self.listings_df.columns:
            if overwrite_existing:
                print("Overwriting existing 'legal_listing' column.")
            else:
                print("'legal_listing' column already exists. Set overwrite_existing=True to overwrite it.")
                return
        self.listings_df['legal_listing'] = self.listings_df['license'].str.match(self.license_regex_pattern, na=False)
        self.listings_df['legal_listing'] = self.listings_df['legal_listing'].astype(bool)
        self.listings_df = self.listings_df.drop(columns=['license'])

        if print_info:
            print("Counts of legal and illegal listings:")
            print(self.listings_df['legal_listing'].value_counts())

    def remove_low_variance_columns(self, threshold=0.1, print_info=True):
        """
        Remove columns with low variance.
        """
        numeric_df = self.listings_df.select_dtypes(include=['float64', 'int64'])
        low_variance_cols = [col for col in numeric_df.columns if numeric_df[col].var() < threshold and col not in self.protected_columns]

        # Remove legal_listing from low variance columns if it exists
        if 'legal_listing' in low_variance_cols:
            low_variance_cols = low_variance_cols.drop('legal_listing')

        if len(low_variance_cols) > 0:
            self.listings_df = self.listings_df.drop(columns=low_variance_cols)
            if print_info:
                print(f"Removed low variance columns: {low_variance_cols}")
        else:
            if print_info:
                print("No low variance columns found to remove.")

    def transform_object_columns(self, print_info=True):
        """
        Transform specific columns for modeling.
        """
        # Convert price to a float variable
        # Check if 'price' column exists and is of object type
        if 'price' not in self.listings_df.columns:
            print("Warning: 'price' column not found in the DataFrame.")
        elif self.listings_df['price'].dtype == 'object':
            self.listings_df['price'] = self.listings_df['price'].str.replace('$', '').str.replace(',', '').astype(float)
        else:
            print("Warning: 'price' column is not of object type, no conversion conducted.")

        # Convert 'host_acceptance_rate' to a float variable
        if 'host_acceptance_rate' not in self.listings_df.columns:
            print("Warning: 'host_acceptance_rate' column not found in the DataFrame.")
        elif self.listings_df['host_acceptance_rate'].dtype == 'object':
            self.listings_df['host_acceptance_rate'] = self.listings_df['host_acceptance_rate'].str.replace('%', '').astype(float)
        else:
            print("Warning: 'host_acceptance_rate' column is not of object type, no conversion conducted.")

        # Convert 'host_response_time' to a float variable
        # The reason is a bit far-fetched for range(0,0.25,0.5,0.75,1), just make it easier for regression model operating. 
        # Moreover it does make sense, to some extent
        if 'host_response_time' not in self.listings_df.columns:
            print("Warning: 'host_response_time' column not found in the DataFrame.")
        elif self.listings_df['host_response_time'].dtype == 'object':
            self.listings_df['host_response_time'] = self.listings_df['host_response_time'].map({
                'within an hour': 1, 'within a few hours': 0.75, 'within a day': 0.5, 'a few days or more': 0.25}).fillna(0)
        else:
            print("Warning: 'host_response_time' column is not of object type, no conversion conducted.")

        # Convert 'host_response_rate' to a float variable
        if 'host_response_rate' not in self.listings_df.columns:
            print("Warning: 'host_response_rate' column not found in the DataFrame.")
        elif self.listings_df['host_response_rate'].dtype == 'object':
            self.listings_df['host_response_rate'] = self.listings_df['host_response_rate'].str.replace('%', '').astype(float)
        else:
            print("Warning: 'host_response_rate' column is not of object type, no conversion conducted.")

        # Convert 'host_verifications' to a float variable
        if 'host_verifications' not in self.listings_df.columns:
            print("Warning: 'host_verifications' column not found in the DataFrame.")
        elif self.listings_df['host_verifications'].dtype == 'object':
            self.listings_df['host_verifications'] = self.listings_df['host_verifications'].map({
                "['email', 'phone', 'photographer', 'work_email']": 1, "['email', 'phone', 'work_email']": 0.75, 
                "['email', 'phone']": 0.5, "['phone', 'work_email']":0.5, 
                "['phone']": 0.25, "['email']": 0.25}).fillna(0)
        else:
            print("Warning: 'host_verifications' column is not of object type, no conversion conducted.")

        # Convert 'host_is_superhost' to a bool variable
        if 'host_is_superhost' not in self.listings_df.columns:
            print("Warning: 'host_is_superhost' column not found in the DataFrame.")
        elif self.listings_df['host_is_superhost'].dtype == 'object':
            self.listings_df['host_is_superhost'] = self.listings_df['host_is_superhost'].map({'t': 1, 'f': 0})
        else:
            print("Warning: 'host_is_superhost' column is not of object type, no conversion conducted.")

        # Convert 'host_has_profile_pic' to a bool variable
        if 'host_has_profile_pic' not in self.listings_df.columns:
            print("Warning: 'host_has_profile_pic' column not found in the DataFrame.")
        elif self.listings_df['host_has_profile_pic'].dtype == 'object':
            self.listings_df['host_has_profile_pic'] = self.listings_df['host_has_profile_pic'].map({'t': 1, 'f': 0})
        else:
            print("Warning: 'host_has_profile_pic' column is not of object type, no conversion conducted.")

        # Convert 'instant_bookable' to a bool variable
        if 'instant_bookable' not in self.listings_df.columns:
            print("Warning: 'instant_bookable' column not found in the DataFrame.")
        elif self.listings_df['instant_bookable'].dtype == 'object':
            self.listings_df['instant_bookable'] = self.listings_df['instant_bookable'].map({'t': 1, 'f': 0})
        else:
            print("Warning: 'instant_bookable' column is not of object type, no conversion conducted.")

        # Convert 'host_identity_verified' to a bool variable
        if 'host_identity_verified' not in self.listings_df.columns:
            print("Warning: 'host_identity_verified' column not found in the DataFrame.")
        elif self.listings_df['host_identity_verified'].dtype == 'object':
            self.listings_df['host_identity_verified'] = self.listings_df['host_identity_verified'].map({'t': 1, 'f': 0})
        else:
            print("Warning: 'host_identity_verified' column is not of object type, no conversion conducted.")

    def one_hot_encode_columns(self, object_columns_list=None, print_info=True):
        """
        One-hot encode categorical columns.
        """
        if object_columns_list is None:
            object_columns_list = self.listings_df.select_dtypes(include='object')
            object_columns_list = list(object_columns_list.columns)
            object_columns_list

        if print_info:
            print("Dropped categories:")

        for colname in object_columns_list:
                # convert column to 'category' dtype
                self.listings_df[colname] = self.listings_df[colname].astype('category')

                # Since we will be dropping the first category of each column, 
                # lets print out the first category of each column so we know what we are dropping
                if print_info:
                    print(colname, ':', self.listings_df[colname].cat.categories[0])

                # applying one-hot coding (drop_first means eliminate one freedom degree to prevent multicollinearity)
                one_hot_encoded = pd.get_dummies(self.listings_df[colname], prefix=colname, drop_first=True)
                # join new columns back to DataFrame
                self.listings_df = self.listings_df.join(one_hot_encoded)

    def remove_separation(self, threshold=1.0, print_info=True):
        """
        Detect and remove variables that cause separation in the logistic regression model.
        """
        X = self.listings_df.drop(columns=['legal_listing'])
        X = pd.get_dummies(X, drop_first=True)  # Convert categorical variables to dummy variables
        y = self.listings_df['legal_listing']

        problematic_cols = detect_separation(X, y, threshold=threshold)
        
        if print_info:
            print("Problematic columns causing separation:", problematic_cols)

        # Drop problematic columns
        self.listings_df = self.listings_df.drop(columns=problematic_cols)

    def remove_missing_values(self, print_info=True):
        """
        First, count the number of NaN values in each column.
        Then, drop columns with more than 10% missing values.
        If print_info is True, print the columns that were dropped.
        Then, drop rows with any NaN values.
        """
        # Count the number of NaN values in each column
        nan_counts = self.listings_df.isna().sum()
        
        # Calculate the threshold for dropping columns (10% of the number of rows)
        threshold = 0.1 * len(self.listings_df)
        
        # Identify columns with more than 10% missing values
        columns_to_drop = [col for col in nan_counts[nan_counts > threshold].index if col not in self.protected_columns]

        
        if print_info and columns_to_drop:
            print(f"Dropping columns with more than 10% missing values: {columns_to_drop}")

        # Drop the identified columns
        self.listings_df = self.listings_df.drop(columns=columns_to_drop)
        
        # Drop rows with any NaN values
        self.listings_df = self.listings_df.dropna()

    def detect_and_remove_multicollinearity(self, vif_threshold=5, print_info=True):
        """
        Detect and remove multicollinear variables using VIF.
        """
        listings_df_VIF = self.listings_df.select_dtypes(include=['bool','float64','int64'])
        listings_df_VIF = listings_df_VIF.astype('float64')

        # calculating VIF

        # Drop all rows containing NAs or infs in listings_df_VIF
        # listings_df_VIF.replace([np.inf, -np.inf], np.nan, inplace=True)
        # listings_df_VIF.dropna(inplace=True)

        listings_df_VIF_new, vif_history, vif_df = drop_column_using_vif_(listings_df_VIF.drop('legal_listing', axis=1), thresh=vif_threshold, print_dropping_columns=print_info)
        
        if print_info:
            print("_" * 50)
            print(f"There are {listings_df_VIF_new.shape[1]} variables after VIF operation.")
            print("_" * 50)
            print("Final VIFs after filtering:")
            print(vif_df)

        # Update the listings_df with the new DataFrame after VIF filtering
        # # Start by getting the columns that were kept
        # kept_columns = listings_df_VIF_new.columns.tolist()
        # # Add 'legal_listing' back to the kept columns
        # kept_columns.append('legal_listing')
        # # Update listings_df with the kept columns
        # self.listings_df = self.listings_df[kept_columns]

        # Add legal_listing back to csv
        listings_df_VIF_new['legal_listing'] = listings_df_VIF['legal_listing']

        # Save into the listings_df variable
        self.listings_df = listings_df_VIF_new.copy()

        # Clean up memory
        del listings_df_VIF, listings_df_VIF_new, vif_history, vif_df

############################################################
# MARK: 2. ASSUMPTION CHECKS
############################################################

    def check_linearity_of_independent_variables_and_log_odds(self, df_city_1=None, print_info=False, show_plot=False):
        """
        Check the linearity of independent variables and log odds.
        Uses the Box-Tidwell test to check for linearity.
        """
        if df_city_1 is None:
            df_city_1 = self.listings_df.copy()
        if df_city_1 is None:
            raise ValueError("Data not loaded. Please load the data before checking linearity.")

        # Begin by getting continous variables
        threshold=3
        continuous_var = []

        for col in df_city_1.columns:
            if pd.api.types.is_numeric_dtype(df_city_1[col]):
                unique_vals = df_city_1[col].nunique(dropna=True)
                if unique_vals >= threshold:
                    continuous_var.append(col)

        # Remove specific columns from continuous_var
        to_remove = ['calculated_host_listings_count_shared_rooms', 'host_total_listings_count', 'bedrooms', 'bathrooms', 'availability_30', 'host_response_time', 'host_verifications', 'beds', 'number_of_reviews_l30d', 'calculated_host_listings_count_entire_homes', 'availability_365', 'number_of_reviews', 'calculated_host_listings_count_private_rooms', 'estimated_occupancy_l365d', 'minimum_nights']
        continuous_var = [col for col in continuous_var if col not in to_remove]

        # Now prep data for Box-Tidwell test
        for var in continuous_var:
        #     # drop values where x = 0
            df_city_1 = df_city_1[df_city_1[var] != 0.0]

            df_city_1[var] = df_city_1[var] + 1e-6  # tiny shift if needed

        for var in continuous_var:
            df_city_1[f'LOGITTRANSFORM_{var}'] = df_city_1[var].apply(lambda x: x * np.log(x)) #np.log = natural log

        # Build the model
        # Keep columns related to continuous variables
        cols_to_keep = continuous_var + df_city_1.columns.tolist()[-len(continuous_var):]
        cols_to_keep

        # Redefine independent variables to include interaction terms
        X_lt = df_city_1[cols_to_keep]
        y_lt = df_city_1['legal_listing']

        X_lt = sm.add_constant(X_lt, prepend=False)

        # Build model and fit the data (using statsmodel's Logit)
        logit_model = sm.Logit(y_lt, X_lt)
        logit_results = logit_model.fit(method='bfgs', maxiter=1000, disp=False)

        # Save the variables that begin with 'LOGITTRANSFORM_' and have a p-value < 0.05
        significant_vars = [var for var in logit_results.params.index if var.startswith('LOGITTRANSFORM_') and logit_results.pvalues[var] < 0.05]

        # Remove the 'LOGITTRANSFORM_' prefix from the variable names
        significant_vars = [var.replace('LOGITTRANSFORM_', '') for var in significant_vars]

        if print_info:
            print(f"Significant variables (p < 0.05) that violate linearity assumption: {significant_vars}")

        return significant_vars


    def check_no_strongly_influential_outliers(self, print_info=False, show_plot=False, remove_hi_outliers=False):
        """
        Check for strongly influential outliers in the dataset.
        Uses Z-score to identify outliers.
        Uses Cook's distance to identify influential outliers.
        """
        if self.listings_df is None:
            raise ValueError("Data not loaded. Please load the data before checking for outliers.")

        X = self.listings_df.drop(columns=['legal_listing'])
        y = self.listings_df['legal_listing']

        # Add constant
        X = sm.add_constant(X, prepend=False)

        # Use GLM method for logreg here so that we can retrieve the influence measures
        logit_model = sm.Logit(y, X)
        logit_results = logit_model.fit(method='bfgs', maxiter=1000)

        # Get influence measures
        influence = logit_results.get_influence()

        # Obtain summary df of influence measures
        summ_df = influence.summary_frame()

        # Filter summary df to Cook distance
        diagnosis_df = summ_df.loc[:,['cooks_d']]

        # Append absolute standardized residual values
        diagnosis_df['std_resid'] = stats.zscore(logit_results.resid_pearson)
        diagnosis_df['std_resid'] = diagnosis_df.loc[:,'std_resid'].apply(lambda x: np.abs(x))

        # Sort by Cook's Distance
        diagnosis_df.sort_values("cooks_d", ascending=False)

        # Set Cook's distance threshold
        cook_threshold = 4 / len(self.listings_df)
        if print_info:
            print(f"Threshold for Cook Distance = {cook_threshold}")

            # How many influential outliers are there?
            influential_outliers = diagnosis_df[(diagnosis_df['cooks_d'] > cook_threshold) & (diagnosis_df['std_resid'] > 3)]
            print(f"Number of influential outliers: {len(influential_outliers)}")

            prop_outliers = round(100*(len(influential_outliers) / len(self.listings_df)),1)
            print(f"Percentage of influential outliers: {prop_outliers}%")

        if show_plot:
            fig, ax = plt.subplots()
            colors = diagnosis_df.apply(lambda row: 'red' if row['std_resid'] > 3 and row['cooks_d'] > cook_threshold else '#1f77b4', axis=1)
            ax.scatter(diagnosis_df['std_resid'], diagnosis_df['cooks_d'], alpha=0.2, c=colors)

            ax.set_xlabel('Standardized Residuals')
            ax.set_ylabel("Cook's Distance")
            ax.set_title("Influential Outliers in Data")
            axhline = ax.axhline(y=cook_threshold, ls="--", color='red')
            axvline = ax.axvline(x=3, ls=":", color='red')

            plt.legend([axhline, axvline], [f'Cook\'s Distance Threshold = 4/N ≈ {cook_threshold:.4f}', 'Standardized Residuals Threshold = 3'], loc='upper right')
            plt.show()

        if remove_hi_outliers:
            self.listings_df = self.listings_df.drop(index=influential_outliers.index)

        

############################################################
# MARK: 3. LOGIT MODEL
############################################################

    def train_model(self, clear_existing_model=False, test_size=0.2, random_state=42, print_info=True):
        """
        Train a logistic regression model.
        """
        if self.logit_model is not None and not clear_existing_model:
            print("Existing model already exists. To clear it, set clear_existing_model=True.")
            return
        elif self.logit_model is not None and clear_existing_model:
            print("Clearing existing model.")

        self.logit_model = LogitModel(target_column='legal_listing', test_size=test_size, random_state=random_state)
        self.logit_model.prepare_data(data=self.listings_df)
        self.logit_model.train_model(print_summary=print_info)

    def evaluate_model(self, selected_threshold=0.5, print_info=True):
        """
        Evaluate the trained model's performance.
        """
        if self.logit_model is None:
            print("No model has been trained yet.")
            return

        self.logit_model.evaluate_model(selected_threshold=selected_threshold)

############################################################
# MARK: 00. XT TOOLS
############################################################


    def does_data_follow_one_in_ten_rule(self, print_info=True):
        """
        Check if the model follows the one-in-ten rule.

        From wikipedia:

        In statistics, the one in ten rule is a rule of thumb for how many predictor parameters can be estimated from data when doing regression analysis (in particular proportional hazards models in survival analysis and logistic regression) while keeping the risk of overfitting and finding spurious correlations low. The rule states that one predictive variable can be studied for every ten events.[1][2][3][4] For logistic regression the number of events is given by the size of the smallest of the outcome categories, and for survival analysis it is given by the number of uncensored events.[3] In other words: for each feature we need 10 observations/labels.

        For example, if a sample of 200 patients is studied and 20 patients die during the study (so that 180 patients survive), the one in ten rule implies that two pre-specified predictors can reliably be fitted to the total data. Similarly, if 100 patients die during the study (so that 100 patients survive), ten pre-specified predictors can be fitted reliably. If more are fitted, the rule implies that overfitting is likely and the results will not predict well outside the training data. It is not uncommon to see the 1:10 rule violated in fields with many variables (e.g. gene expression studies in cancer), decreasing the confidence in reported findings.[5]

        """
        if self.logit_model is None:
            raise ValueError("Model not trained. Please train the model before checking the one-in-ten rule.")

        # Get the number of events (illegal listings)
        num_events = self.listings_df['legal_listing'].value_counts().min()

        # Calculate the maximum number of predictors allowed
        max_predictors = num_events // 10

        # Get the number of predictors in the model
        num_predictors = self.logit_model.X_train.shape[1]

        if print_info:
            print(f"Number of events (legal/illegal listings): {num_events}")
            print(f"Maximum number of predictors allowed (one-in-ten rule): {max_predictors}")
            print(f"Number of predictors in the model: {num_predictors}")

        if num_predictors > max_predictors:
            if print_info:
                print("Warning: The model violates the one-in-ten rule!")
            return False
        else:
            if print_info:
                print("The model follows the one-in-ten rule.")
            return True

    def print_data_summary(self):
        """
        Print a summary of the data.
        """
        print("Data Shape:")
        print(self.listings_df.shape)
        print("First 3 rows of the DataFrame:")
        print(self.listings_df.head(3))
        print("Counts of legal and illegal listings:")
        print(self.listings_df['legal_listing'].value_counts())

    def run_initial_setup(self, print_info=False):
        """
        Run the entire pipeline using default settings.
        """
        self.load_data()
        self.preprocess_data(print_info=print_info)
        self.create_legal_listing_column(print_info=print_info, overwrite_existing=False)
        self.transform_object_columns()
        self.remove_low_variance_columns(print_info=print_info)
        self.remove_missing_values(print_info=print_info)
        self.one_hot_encode_columns(print_info=print_info)
        self.remove_separation(print_info=print_info)
        self.detect_and_remove_multicollinearity(print_info=print_info)

