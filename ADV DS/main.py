import streamlit as st
import pandas as pd
import os
import numpy as np
from PIL import Image
import torch
from baselinemodel import Model 
 # Replace 'your_model_module' with the actual module where the Model class is defined

# Load data
# portfolio_data = pd.read_json("./data/portfolio.json")
# profile_data = pd.read_json("./data/profile.json")
# transcript_data = pd.read_json("./data/transcript.json")

# Load recommendation model (replace with your actual model loading code)

# Force Tesorflow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



# Load your PyTorch model
def load_pytorch_model(model_path):
    model = Model(100, 100,  # Provide the appropriate values for N and M
                  n_cont_user=df[user_specs].shape[1], 
                  n_cont_offer=df[offer_specs].shape[1],
                  embed_dim=D, 
                  output_dim=df['event'].nunique(), 
                  layers=layers)
    
    model.load_state_dict(torch.load(model_path))
    return model

# File upload for model
model_file = st.file_uploader("Upload PyTorch Model", type=["pth"])

if model_file is not None:
    model = load_pytorch_model(model_file)
    st.success("PyTorch Model loaded successfully!")

# Now you can use the 'model' object in the rest of your Streamlit app.


# Project Overview
st.markdown("# Starbucks Offer Recommendation System")
st.markdown("This project aims to optimize the customer experience by predicting and spreading offers that are personalized for each customer.")

# Metrics Description
st.markdown("## Metrics Description")
st.write("Metrics are essential for evaluating the performance of the recommendation model. The confusion matrix and key metrics include:")
# Display the confusion matrix as a table (similar to the example provided in the previous response)

# Model Evaluation
st.markdown("## Model Evaluation")
st.write("Evaluate the performance of different recommendation models. Compare precision, recall, and other relevant metrics.")
# Display charts or visualizations using st.bar_chart, st.line_chart, st.area_chart, etc.


# Personalized Offer Recommendations
st.markdown("## Personalized Offer Recommendations")
# Input fields for user details
age = st.slider("Select Age", min_value=18, max_value=100, value=30)
gender = st.radio("Select Gender", ["Male", "Female", "Other"])
income = st.number_input("Enter Income", min_value=0, value=50000)

# Button to trigger personalized recommendations
if st.button("Get Recommendations"):
    # Use the model to generate and display personalized recommendations
    recommendations = generate_recommendations(model, age, gender, income)  # Replace with your actual recommendation function
    st.write("Your Personalized Offer Recommendations:")
    st.write(recommendations)




