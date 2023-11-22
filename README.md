# yvr-airbnb-license-model
Predicting the legality of Airbnb's in Vancouver using quantitative method models.

## Overview
This project aims to predict the legality of Airbnb's in Vancouver. Using Airbnb data collected from [InsideAirbnb](http://insideairbnb.com/vancouver) (yvr_listing_data.csv), we will use the available quantitative and qualitative data to predict whether or not each individual Airbnb has a license. In essense, we will use quantitative methods to see if there is strong relationship between the legality of the Airbnb, and the variables (or combination of them) associated with the listing.

The repository has 3 main jupyter notebooks:

1. data_cleaning.ipynb: This notebook is used to clean the data and prepare it for analysis. The cleaned data is saved as yvr_listing_data_cleaned.csv.

2. exploratory_analysis.ipynb: This notebook is used to explore the data and find any interesting relationships between the variables.

3. model.ipynb: This notebook also contains the code for the models used to predict the legality of the Airbnb's.

TODO:
- Exploratory analysis: What data columns make sense for us to use an independent variables?
- Remove data for long-term rentals (>= 30 days minimum) since they don't require a license.
- We'll likely end up doing a large multiple regression model. Couple of things we need to consider:
      - How do we deal with categorical data?
      - How do we ensure there is minimal multicollinearity between independant variables?
