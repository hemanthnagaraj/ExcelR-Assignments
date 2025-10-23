#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor - Model Comparison",
    layout="wide")

# App title
st.title("Titanic Survival Prediction - Model Comparison")
st.markdown("""
Compare two logistic regression models with different preprocessing:
- **Model 01**: Trained on unscaled data
- **Model 02**: Trained on scaled data
""")

model_01_coefficients = {
    'Pclass': -65.9401, 'SibSp': -27.7586, 'Fare': 0.178, 
    'Age_imput': -3.8088, 'Sex_male': -92.5664, 
    'Embarked_Q': -0.4098, 'Embarked_S': -32.3184}

model_02_coefficients = {
    'Pclass': -59.7696, 'SibSp': -31.0883, 'Fare': 8.5997, 
    'Age_imput': -39.1587, 'Sex_male': -71.8753, 
    'Embarked_Q': -1.0499, 'Embarked_S': -16.8186}

# The coefficients are obtained from model.coef_ in the Logistic_Regression.ipynb
def calculate_survival_probability(input_features, model_type = 'scaled'):
    """
    Calculate survival probability based on calculated coefficients
    """
    # Actual coefficients from the Logistic_Regression
    # Unscaled model
    if model_type == 'unscaled':
        coefficients = model_01_coefficients
    else:  # Scaled model
        coefficients = model_02_coefficients
    
    # Unpack input features
    pclass, sibsp, fare, age, sex_male, embarked_q, embarked_s = input_features
    
    # Calculate log-odds
    log_odds = (
        coefficients['Pclass'] * (pclass - 2) +  # Center Pclass
        coefficients['SibSp'] * sibsp +
        coefficients['Fare'] * (fare - 32) / 10 +  # Scale fare
        coefficients['Age_imput'] * (age - 30) / 10 +  # Scale age
        coefficients['Sex_male'] * sex_male +
        coefficients['Embarked_Q'] * embarked_q +
        coefficients['Embarked_S'] * embarked_s -
        0.5  )# Intercept adjustment
    
    # Convert to probability
    probability = 1 / (1 + np.exp(-log_odds))
    return probability

def main():
    # Sidebar for user input
    st.sidebar.header("Passenger Information")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], 
                             help="1 = 1st Class (Upper), 2 = 2nd Class (Middle), 3 = 3rd Class (Lower)")
        age = st.slider("Age", 0.0, 80.0, 30.0, 1.0) # min, max, avg, increment
        sex = st.selectbox("Sex", ["Female", "Male"])
        embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"], 
                               help = "C = Cherbourg, Q = Queenstown, S = Southampton")
    
    with col2:
        sibsp = st.slider("Siblings/Spouses Aboard", 0, 8, 0)
        fare = st.slider("Fare", 0.0, 263.0, 33.0, 0.5) # min, max, avg, increment 
    
    # Convert categorical variables to model format
    sex_male = 1 if sex == "Male" else 0
    embarked_q = 1 if embarked == "Q" else 0
    embarked_s = 1 if embarked == "S" else 0
    
    input_features = [pclass, sibsp, fare, age, sex_male, embarked_q, embarked_s]
    
    # Make predictions
    if st.sidebar.button("Predict Survival", type = "primary"):
        
        # Calculate probabilities for both models
        prob_01 = calculate_survival_probability(input_features, 'unscaled')
        prob_02 = calculate_survival_probability(input_features, 'scaled')
        
        # Display predictions
        st.header("Survival Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model 01 (Unscaled Data)")
            st.metric(
                "Survival Probability", 
                f"{prob_01:.1%}",
                f"Prediction: {'Survived' if prob_01 > 0.5 else 'Did Not Survive'}")
            
            # Show probability breakdown
            fig1 = go.Figure(data = [
                go.Bar(x = ['Did Not Survive', 'Survived'], 
                      y = [(1-prob_01)*100, prob_01*100],
                      marker_color = ['#EF553B', '#00CC96'])])
            fig1.update_layout(title = "Probability Distribution", yaxis_title = "Probability (%)")
            st.plotly_chart(fig1, use_container_width = True)
        
        with col2:
            st.subheader("Model 02 (Scaled Data)")
            st.metric(
                "Survival Probability", 
                f"{prob_02:.1%}",
                f"Prediction: {'Survived' if prob_02 > 0.5 else 'Did Not Survive'}")
            
            # Show probability breakdown
            fig2  =  go.Figure(data = [
                go.Bar(x = ['Did Not Survive', 'Survived'], 
                      y = [(1-prob_02)*100, prob_02*100],
                      marker_color = ['#EF553B', '#00CC96'])])
            fig2.update_layout(title = "Probability Distribution", yaxis_title = "Probability (%)")
            st.plotly_chart(fig2, use_container_width = True)
        
        # Show agreement between models
        st.subheader("Model Agreement")
        prediction_01 = "Survived" if prob_01 > 0.5 else "Did Not Survive"
        prediction_02 = "Survived" if prob_02 > 0.5 else "Did Not Survive"
        
        if prediction_01 == prediction_02:
            st.success(f"Models Agree: Passenger would **{prediction_01}**")
        else:
            st.warning(f"⚠️ Models Disagree: Model 01 predicts **{prediction_01}**, Model 02 predicts **{prediction_02}**")
    
    # Model Comparison Section
    st.header("Model Coefficient Comparison")
    
    # Create comparison chart
    features = list(model_01_coefficients.keys())
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name = 'Model 01 (Unscaled)',
        x = features,
        y = [model_01_coefficients[f] for f in features],
        marker_color = 'blue',
        text = [f"{model_01_coefficients[f]:.1f}%" for f in features],
        textposition = 'auto'))
    
    fig.add_trace(go.Bar(
        name = 'Model 02 (Scaled)',
        x = features,
        y = [model_02_coefficients[f] for f in features],
        marker_color = 'red', 
        text = [f"{model_02_coefficients[f]:.1f}%" for f in features],
        textposition = 'auto'))
    
    fig.update_layout(
        title = "Percentage Impact on Survival Odds by Feature",
        xaxis_title = "Features",
        yaxis_title = "Percentage Impact on Odds",
        barmode = 'group',
        height = 500)
    
    st.plotly_chart(fig, use_container_width = True)
    
    # Key Insights
    st.header("Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model 01 (Unscaled)")
        st.write("""
        - **Age**: Minimal impact (-3.8%)
        - **Fare**: Negligible impact (+0.2%)
        - **Sex**: Extreme male penalty (-92.6%)
        - **Pclass**: Strong negative impact
        """)
    
    with col2:
        st.subheader("Model 02 (Scaled)") 
        st.write("""
        - **Age**: Significant impact (-39.2%)
        - **Fare**: Positive impact (+8.6%)
        - **Sex**: Moderate male penalty (-71.9%)
        - **Pclass**: Strong but reduced impact
        """)
    
    st.info("""
    **Conclusion**: Model 02 (scaled data) provides more reasonable feature impacts about the Titanic disaster.
    Age and fare have more meaningful impacts, and the male survival penalty is less extreme.
    """)

if __name__ == '__main__':
    main()

