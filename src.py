
"""
Helper functions for the project.
"""

from statsmodels.stats.outliers_influence import variance_inflation_factor 
from statsmodels.tools.tools import add_constant
import pandas as pd

def drop_column_using_vif_(df, thresh=5):
    '''
    This function is adjusted from: https://stackoverflow.com/a/51329496/4667568

    Calculates VIF each feature in a pandas dataframe, and repeatedly drop the columns with the highest VIF
    A constant must be added to variance_inflation_factor or the results will be incorrect

    :param df: the pandas dataframe containing only the predictor features, not the response variable
    :param thresh: (default 5) the threshould VIF value. If the VIF of a variable is greater than thresh, it should be removed from the dataframe
    :return: dataframe with multicollinear features removed
    '''
    while True:
        
        # adding a constatnt item to the data. add_constant is a function from statsmodels (see the import above)
        df_with_const = add_constant(df,has_constant='add')

        if 'const' in df_with_const.columns:
            vif_df = pd.Series([variance_inflation_factor(df_with_const.values, i) for i in range(df_with_const.shape[1])], name= "VIF", 
                                index=df_with_const.columns).to_frame()

            # drop the const
            vif_df = vif_df.drop('const')
        else:
            raise ValueError("constant column 'const' not successfully added")
        
        # if the largest VIF is above the thresh, remove a variable with the largest VIF
        # If there are multiple variabels with VIF>thresh, only one of them is removed. This is because we want to keep as many variables as possible
        if vif_df.VIF.max() > thresh:
            # If there are multiple variables with the maximum VIF, choose the first one
            index_to_drop = vif_df.index[vif_df.VIF == vif_df.VIF.max()].tolist()[0]
            print('Dropping: {} (VIF: {})'.format(index_to_drop, vif_df.loc[index_to_drop, 'VIF']))
            df = df.drop(columns = index_to_drop)
        else:
            # No VIF is above threshold. Exit the loop
            break

    return df


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