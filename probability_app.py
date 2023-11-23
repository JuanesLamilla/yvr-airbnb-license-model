"""
Using Streamlit, create a web app that takes in a user input of short-term rental data, 
and returns the probability that the rental is legal (licensed) or not.

This app uses the probability formula determined by the 
multiple logistic regression model in model.ipynb.
"""

# Import libraries
import streamlit as st
import pandas as pd

# Model constants
CONT = -1.0386
REVIEW_COEF = 0.2498
PRICE_COEF = 0.0044
INSTBOOK_COEF = 0.2316

# Create streamlit app
st.title('Short-Term Rental License Probability Calculator: Vancouver')
st.write('This app uses a multiple logistic regression model to calculate the probability that a short-term rental in Vancouver is licensed or not. The model was trained on data from the InsideAirbnb.')

# Ask the user for the rental's information
st.header('Input the following information about the rental:')
review_score = st.slider('Review Score (out of 5)', min_value=0.0, max_value=5.0)
price = st.number_input('Price (in CAD)', min_value=0.00, step=1.00)
instant_book = st.selectbox('Instantly Bookable?', ['Yes', 'No'])

# Convert instant_book to binary
if instant_book == 'Yes':
    instant_book = 1
else:
    instant_book = 0

# Calculate the probability that the rental is licensed
prob = 1 / (1 + 2.71828 ** -(CONT + (REVIEW_COEF * review_score) + (PRICE_COEF * price) + (INSTBOOK_COEF * instant_book)))

# Display the probability as a percentage.
st.header('The probability that this rental is licensed is:')
if prob < 0.5:
    st.title(':red[' + str(round(prob * 100, 2)) + '%]')
elif prob < 0.75:
    st.title(':orange[' + str(round(prob * 100, 2)) + '%]')
else:
    st.title(':green[' + str(round(prob * 100, 2)) + '%]')
