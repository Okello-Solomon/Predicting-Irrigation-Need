# Predicting Irrigation Need Using XGBoost

**Live Application:**

https://your-streamlit-app-link-here

Use the link above to interact with the application and predict the level of irrigation required (Low, Medium, High) in real time based on environmental, soil, and crop conditions.

## Project Overview

Efficient irrigation is a critical component of modern agriculture. Both over-irrigation and under-irrigation can lead to reduced crop yields, resource wastage, and environmental degradation. With the rise of data-driven technologies, machine learning offers a powerful approach to optimize irrigation decisions.

This project develops a predictive machine learning model that classifies irrigation needs using key environmental, soil, and crop-related features.

<details> <summary><strong>View Project Details</strong></summary>

**The system predicts irrigation requirements at three levels:**

- Low Irrigation Need
- Medium Irrigation Need
- High Irrigation Need

By analyzing factors such as soil moisture, temperature, rainfall, crop growth stage, and farming practices like mulching, the model provides accurate and real-time irrigation recommendations.

**This project demonstrates a complete end-to-end machine learning workflow:**

data preprocessing → feature engineering → model training → model evaluation → deployment via Streamlit

## Features
Predicts irrigation needs in real time
Supports multiclass classification (Low, Medium, High)
Utilizes environmental, soil, and crop-related features
Handles categorical and numerical data effectively
Provides probability-based predictions for better decision-making
Interactive and user-friendly Streamlit interface
Enables data-driven irrigation planning for improved efficiency
## Machine Learning Model

**Model Used:** XGBoost Classifier

**Objective:** Predict irrigation need using selected key agricultural features

**Target Classes:**

0 → Low

1 → Medium

2 → High

## Evaluation Metrics:
The model was evaluated using cross-validation and classification metrics such as accuracy, precision, recall, and F1-score to ensure reliable and balanced performance across all classes, including the minority class (High irrigation).

## Why XGBoost Was Selected

🎯 Business Goals
Optimize water usage in agriculture
Reduce over-irrigation and under-irrigation
Improve crop yield and health
Support precision agriculture and smart farming systems
Enable data-driven decision-making for farmers and stakeholders
📌 Objectives
- Develop a robust machine learning model to predict irrigation needs
- Identify key factors influencing irrigation requirements
- Minimize misclassification, especially for high irrigation needs
- Build an interactive system for real-time irrigation prediction
- Support sustainable and climate-smart agricultural practices

## Data Description

**Source:**
The dataset was obtained from the Kaggle Playground Series (2026). It is a synthetic dataset generated using a deep learning model trained on a real-world irrigation prediction dataset.

**[Predicting Irrigation Need Dataset](https://www.kaggle.com/competitions/playground-series-s6e4/data)**

## Deployment

The model is deployed using Streamlit, allowing users to:

Input environmental and crop conditions
Receive real-time irrigation predictions
View probability distributions for each irrigation level
Make informed irrigation decisions

## Conclusion

This project demonstrates how machine learning can transform irrigation management. By leveraging environmental and agricultural data, the XGBoost model provides highly accurate predictions that can help reduce water wastage, improve crop yield, and promote sustainable farming practices.