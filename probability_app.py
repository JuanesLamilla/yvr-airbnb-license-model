"""
Using Streamlit, create a web app that takes in a user input of short-term rental data, 
and returns the probability that the rental is legal (licensed) or not.

This app uses the probability formula determined by the 
multiple logistic regression model in sm_model.ipynb.
"""

# Import libraries
import streamlit as st

# Model constants
CONT = -1.9937601640705742
NUM_REVIEW_COEF = 0.03364956476964472
PRICE_COEF = 0.0071342501459958005
IS_PRIVATE_ROOM_COEF = -0.8774995083971812
NEIGHBOURHOOD_COEFS = {"Downtown Eastside": 0.7727896964515187, "Dunbar Southlands": 1.589589473246962,
                       "Grandview-Woodland": 1.3927171041689204, "Hastings-Sunrise": 2.1308338849465738,
                       "Kensington-Cedar Cottage": 1.8828917971409502, "Killarney": 1.740440987902061,
                       "Marpole": 0.8736770236664401, "Mount Pleasant": 0.5947986707412933,
                       "Renfrew-Collingwood": 1.7689101917560193, "Riley Park": 1.059405258158975,
                       "Shaughnessy": 2.001705602743299, "Strathcona": 2.1377709297563325,
                       "Sunset": 1.8735559819953822, "Victoria-Fraserview": 2.4681859816342926,
                       "West End": -0.6313713841819086, "West Point Grey": 1.703693351953927,
                       "Other": 0.0}

# Create streamlit app
st.title('Short-Term Rental License Probability Calculator: Vancouver')
st.write('This app uses a multiple logistic regression model to calculate the probability that a short-term rental in Vancouver is licensed or not. The model was trained on data from InsideAirbnb.')

# Ask the user for the rental's information
st.header('Input the following information about the rental:')
num_reviews = st.number_input('Number of Reviews', min_value=0, step=1)
price = st.number_input('Price (in CAD)', min_value=0.00, step=1.00)
is_private_room = st.radio('Is the rental a private room?', ('Yes', 'No'))
neighbourhood = st.selectbox('Neighbourhood', list(NEIGHBOURHOOD_COEFS.keys()))

# Convert instant_book to binary
if is_private_room == 'Yes':
    is_private_room = 1
else:
    is_private_room = 0

# Calculate the probability that the rental is licensed
prob = 1 / (1 + 2.71828 ** -(CONT + (NUM_REVIEW_COEF * num_reviews) + (PRICE_COEF * price) + (IS_PRIVATE_ROOM_COEF * is_private_room) + (NEIGHBOURHOOD_COEFS[neighbourhood])))

# Display the probability as a percentage.
st.header('The probability that this rental is licensed is:')
if prob < 0.5:
    st.title(':red[' + str(round(prob * 100, 2)) + '%]')
elif prob < 0.75:
    st.title(':orange[' + str(round(prob * 100, 2)) + '%]')
else:
    st.title(':green[' + str(round(prob * 100, 2)) + '%]')


st.markdown("""---""")

st.write("Created and modelled by [Juanes Lamilla](http://juaneslamilla.github.io), [Bohao Su](https://github.com/BohaoSuCC), [Nikhil Desai](https://github.com/NikhilSDesai), and [Simon Song](https://github.com/songzimen) for the MSc in Urban Spatial Science at UCL Centre for Advanced Spatial Analysis.")

st.write("Using just these variables, this model is able to predict if a short-term rental in Vancouver is licensed or not with an accuracy of 85%.")

st.write("Learn more about the model on the [GitHub repository](https://github.com/JuanesLamilla/yvr-airbnb-license-model).")