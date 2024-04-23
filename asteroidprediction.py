import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
#import plotly.express as px
 
# Load the model
model = joblib.load('/workspaces/Data-Science-Methods-and-Tools/RandomForest_final.pkl')
 
# Function to predict hazardous status and probabilities
def predict_hazardous(features):
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    return prediction[0], probability[0]  # Return both prediction and probability
 
# Function to create features dataframe
def create_features(user_inputs):
    features = pd.DataFrame([user_inputs])
    return features
 
# Function to plot feature importance
def plot_feature_importance(model, df):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=df.columns[indices], palette='viridis', ax=ax)
    plt.title('Feature Importances')
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    plt.xticks(rotation=45)
    st.pyplot(fig)
 
# Function to plot probability bar chart using Plotly
# def plot_probability_chart(probability):
#     data = pd.DataFrame({
#         'Category': ['Non-Hazardous', 'Hazardous'],
#         'Probability': probability
#     })
#     fig = px.bar(data, x='Category', y='Probability', title="Probability of Hazardous Status",
#                  labels={'Probability': 'Probability', 'Category': 'Status'},
#                  color='Category', color_discrete_map={'Non-Hazardous': 'blue', 'Hazardous': 'red'})
#     st.plotly_chart(fig)
 
# Custom Streamlit configuration
st.set_page_config(page_title='Asteroid Hazard Prediction', layout='wide')
st.markdown("""
<style>
    .big-font {
        font-size:300% !important;
    }
</style>
""", unsafe_allow_html=True)
 
# Streamlit UI
st.title('Asteroid Hazard Prediction', anchor=None)
 
# Sidebar with user input features
st.sidebar.title('User Input Features')
user_inputs = {}
 
# Define the ranges and tooltip text for each feature
feature_info = {
    'Absolute Magnitude': ((0.0, 50.0), 'Absolute magnitude of the asteroid (0.0 - 50.0)'),
    'Est Dia in Miles(max)': ((0.0, 25.0), 'Estimated diameter of the asteroid in miles (0.0 - 25.0)'),
    'Miles per hour': ((0.0, 10000.0), 'Velocity of the asteroid in miles per hour (0.0 - 10000.0)'),
    'Miss Dist.(miles)': ((0.0, 90000000.0), 'Miss distance of the asteroid in miles (0.0 - 90000000.0)'),
    'Orbit Uncertainity': ((0, 9), 'Orbit uncertainity (0 - 9)'),
    'Minimum Orbit Intersection': ((0.0, 0.6), 'Minimum orbit intersection (0.0 - 0.6)'),
    'Eccentricity': ((0.0, 1.0), 'Eccentricity of the asteroid (0.0 - 1.0)'),
    'Inclination': ((0.0, 80.0), 'Inclination of the asteroid in degrees (0.0 - 80.0)'),
    'Asc Node Longitude': ((0.0, 360.0), 'Ascending node longitude of the asteroid in degrees (0.0 - 360.0)'),
    'Orbital Period': ((0.0, 4200.0), 'Orbital period of the asteroid (0.0 - 4200.0)'),
    'Perihelion Distance': ((0.0, 1.30), 'Perihelion distance of the asteroid in AU (0.0 - 1.30)'),
    'Perihelion Arg': ((0.0, 360.0), 'Perihelion argument of the asteroid in degrees (0.0 - 360.0)'),
    'Aphelion Dist': ((0.0, 9.0), 'Aphelion distance of the asteroid in AU (0.0 - 9.0)'),
    'Perihelion Time': ((0.0, 250000.0), 'Perihelion time of the asteroid (0.0 - 250000.0)'),
    'Mean Anomaly': ((0.0, 360.0), 'Mean anomaly of the asteroid in degrees (0.0 - 360.0)'),
    'Mean Motion': ((0.0, 2.2), 'Mean motion of the asteroid (0.0 - 2.2)'),
}
 
# Create input fields for each feature with tooltip text
for feature, ((min_val, max_val), tooltip) in feature_info.items():
    step_val = 1 if isinstance(min_val, int) and isinstance(max_val, int) else 0.01
    user_inputs[feature] = st.sidebar.number_input(f'Enter {feature}', min_value=min_val, max_value=max_val, value=min_val, step=step_val, help=tooltip)
 
# Button to predict hazardous status
if st.sidebar.button('Predict Hazardous Status'):
    features = create_features(user_inputs)
    result, probability = predict_hazardous(features)
    status = 'hazardous' if result == 1 else 'not hazardous'
    if result == 1:
        st.sidebar.error(f'This asteroid is {status}!', icon="⚠️")
    else:
        st.sidebar.success(f'This asteroid is {status}!', icon="✅")
    st.sidebar.write(f"Probability of being non-hazardous: {probability[0]:.2f}")
    st.sidebar.write(f"Probability of being hazardous: {probability[1]:.2f}")
 
    # Option to display the probability bar chart
    # if st.sidebar.checkbox('Show Probability Chart'):
    #     plot_probability_chart(probability)
 
# Image of an asteroid
st.image('/workspaces/Data-Science-Methods-and-Tools/asteroid_img (1).jpg', use_column_width=True, caption='Image of an asteroid')
 
 
show_graphs = st.checkbox('Show Graphs')
 
# Generate graphs if the checkbox is checked
if show_graphs:
    st.header('Graphs')
    features = create_features(user_inputs)  # Create features dataframe here
    # Plot feature importance
    st.subheader('Feature Importance')
    plot_feature_importance(model, features)
 
    # # Plot probability chart
    # st.subheader('Probability Chart')
    # _, probability = predict_hazardous(features)
    # plot_probability_chart(probability)
 
   
# Additional feature information section
with st.expander("Learn more about asteroid features"):
    st.write("""
    Here you can find more information about the features used to predict asteroid hazards:
    - **Absolute Magnitude:** The brightness of an asteroid if it were placed at a standard distance.
    - **Estimated Diameter:** The approximate size of the asteroid.
    - **Velocity:** The speed at which the asteroid is moving through space.
    - **Miss Distance:** How close the asteroid will come to Earth.
    - **Orbit Uncertainty:** A measure of the uncertainty in the asteroid's orbital parameters.
    - **Minimum Orbit Intersection:** The minimum distance between the asteroid's orbit and Earth's orbit.
    - **Eccentricity:** A measure of how elliptical the asteroid's orbit is.
    - **Inclination:** The tilt of the asteroid's orbit relative to Earth's orbital plane.
    - **Ascending Node Longitude:** The location where the asteroid's orbit crosses Earth's orbital plane, measured in degrees.
    - **Orbital Period:** The time it takes for the asteroid to complete one orbit around the Sun.
    - **Perihelion Distance:** The closest distance between the asteroid and the Sun in its orbit, measured in Astronomical Units (AU).
    - **Perihelion Argument:** The angle between the asteroid's perihelion and the ascending node, measured in degrees.
    - **Aphelion Distance:** The farthest distance between the asteroid and the Sun in its orbit, measured in Astronomical Units (AU).
    - **Perihelion Time:** The time at which the asteroid is closest to the Sun.
    - **Mean Anomaly:** An angular parameter that defines the current position of the asteroid along its orbit.
    - **Mean Motion:** The average angular speed of the asteroid along its orbit.
    """)
 
# Ensure the web app is mobile-responsive
st.markdown("""
<style>
    @media (max-width: 768px) {
        .sidebar .sidebar-content {
            padding: 0 !important;
        }
    }
</style>
""", unsafe_allow_html=True)
