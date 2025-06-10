
"""
Helper functions for the project.
"""

from statsmodels.stats.outliers_influence import variance_inflation_factor 
from statsmodels.tools.tools import add_constant
import pandas as pd

def drop_column_using_vif_(df, thresh=5, print_dropping_columns=True, exclude_columns=None):
    '''
    This function is adjusted from: https://stackoverflow.com/a/51329496/4667568

    Calculates VIF each feature in a pandas dataframe, and repeatedly drop the columns with the highest VIF, while protecting exclude_columns
    A constant must be added to variance_inflation_factor or the results will be incorrect

    :param df: the pandas dataframe containing only the predictor features, not the response variable
    :param thresh: (default 5) the threshould VIF value. If the VIF of a variable is greater than thresh, it should be removed from the dataframe
    :return: dataframe with multicollinear features removed
    '''
    exclude_columns = exclude_columns or []
    vif_history = []  # save the list for VIFs of each iteration

    while True:
        
        # adding a constatnt item to the data. add_constant is a function from statsmodels (see the import above)
        df_with_const = add_constant(df,has_constant='add')

        if 'const' in df_with_const.columns:
            vif_df = pd.Series(
                [variance_inflation_factor(df_with_const.values, i) for i in range(df_with_const.shape[1])],
                name="VIF",
                index=df_with_const.columns
            ).to_frame()

            vif_df = vif_df.drop('const')
            vif_history.append(vif_df.copy())
        else:
            raise ValueError("constant column 'const' not successfully added")
        
        candidates = vif_df[(vif_df['VIF'] > thresh) & (~vif_df.index.isin(exclude_columns))]
        
        # if the largest VIF is above the thresh, remove a variable with the largest VIF
        # If there are multiple variabels with VIF>thresh, only one of them is removed. This is because we want to keep as many variables as possible
    
        if not candidates.empty:
            # 只删除一个 VIF 最大的变量
            index_to_drop = candidates['VIF'].idxmax()
            if print_dropping_columns:
                print(f"Dropping: {index_to_drop} (VIF: {vif_df.loc[index_to_drop, 'VIF']})")
            df = df.drop(columns=index_to_drop)
        else:
            # No VIF is above threshold. Exit the loop
            break

    # print("Final VIFs after filtering:")
    # print(vif_df)

    return df, vif_history, vif_df


def show_vif_values(df, dependent_variable):
    """
    Takes a dataframe and the name of the dependent variable, 
    and returns a dataframe with the VIF values for each column (independent variables).
    """
    # Exclude the dependent_variable column from the analysis
    df.drop(columns=dependent_variable, inplace=True)

    df_with_const = add_constant(df,has_constant='add')

    vif_df = pd.Series([variance_inflation_factor(df_with_const.values, i) for i in range(df_with_const.shape[1])], name= "VIF", 
                            index=df_with_const.columns).to_frame()

    vif_df = vif_df.drop('const')

    # Sort the dataframe by VIF values in descending order
    vif_df = vif_df.sort_values(by='VIF', ascending=False)

    return vif_df

def detect_separation(X, y, threshold=1.0):
    """
    Checks for variables in X that can perfectly or nearly perfectly predict y.
    
    Parameters:
    - X: pd.DataFrame of predictors
    - y: pd.Series of binary target (0/1)
    - threshold: proportion (default=1.0) for 'perfect' separation. Use <1.0 to check for near-separation.
    
    Returns:
    - List of column names likely causing separation
    """
    problematic_cols = []

    for col in X.columns:
        # if X[col].nunique() > 50:
        if not isinstance(X[col], pd.CategoricalDtype) or X[col].nunique() > 50:
            continue  # Skip continuous variables for now

        cross_tab = pd.crosstab(X[col], y, normalize='index')

        for val in cross_tab.index:
            # Look for cases where a category always maps to one outcome
            if (cross_tab.loc[val] == 1).any() or (cross_tab.loc[val] == 0).any():
                max_class_prop = cross_tab.loc[val].max()
                if max_class_prop >= threshold:
                    problematic_cols.append(col)
                    break  # No need to check other values in this column

    return list(set(problematic_cols))