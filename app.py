import streamlit as st
import pandas as pd
import joblib

# --- Page Config ---
st.set_page_config(
    page_title="Irrigation Prediction",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model ---
pipeline = joblib.load("xgb_irrigation_model.pkl")

# --- Sidebar Tabs ---
import streamlit as st

# Inject stronger CSS
st.markdown("""
<style>

/* Sidebar title */
.sidebar-title {
    font-size: 22px;
    font-weight: 700;
    color: blue;
    margin-bottom: 0px;
}

/* Radio labels (Prediction, Report) */
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 18px !important;
    font-weight: 700 !important;
    color: blue !important;
}

/* Selected option */
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p {
    color: darkblue !important;
    font-weight: 800 !important;
}

</style>
""", unsafe_allow_html=True)

# Custom title
st.sidebar.markdown('<div class="sidebar-title">Select Section</div>', unsafe_allow_html=True)

# Radio
menu = st.sidebar.radio("", ["Prediction", "Report"])

# =========================
# --- PREDICTION TAB ---
# =========================
if menu == "Prediction":
    # --- App Title ---
    st.markdown("<h1 style='color:green'>🌱 Irrigation Need Prediction</h1>", unsafe_allow_html=True)
    st.markdown("""
    Predict the **level of irrigation required** for a farm.

    **Classes:**  
    0 = Low  
    1 = Medium  
    2 = High
    """)

    # --- Sidebar Inputs ---
    st.sidebar.markdown("<h2 style='color:green'>🌾 Farm Conditions</h2>", unsafe_allow_html=True)

    Soil_Moisture = st.sidebar.slider("Soil Moisture (%)", 0.0, 100.0, 37.0)
    Temperature_C = st.sidebar.slider("Temperature (°C)", 12.0, 42.0, 27.0)
    Humidity = st.sidebar.slider("Humidity (%)", 25.0, 95.0, 62.0)
    Rainfall_mm = st.sidebar.number_input("Rainfall (mm)", 0.38, 2499.69, 1462.2)
    Wind_Speed_kmh = st.sidebar.number_input("Wind Speed (km/h)", 0.5, 20.0, 10.38)
    Previous_Irrigation_mm = st.sidebar.number_input("Previous Irrigation (mm)", 0.02, 119.99, 62.32)

    water = st.sidebar.selectbox("Water Source", ['Rainwater','Reservoir','River','Groundwater'])

    # --- Main Panel: Crop & Irrigation Setup ---
    st.markdown("<h3 style='color:green'>🌾 Crop & Irrigation Setup</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        Mulching_Used = st.selectbox("Mulching Used", ["Yes", "No"])
        Mulching_Used = 1 if Mulching_Used == "Yes" else 0
        stage_map = {'Sowing':1, 'Vegetative':2, 'Flowering':3, 'Harvest':4}
        Crop_Growth_Stage = st.selectbox("Crop Growth Stage", list(stage_map.keys()))
        Crop_Growth_Stage = stage_map[Crop_Growth_Stage]

    with col2:
        crop = st.selectbox("Crop Type", ['Maize','Potato','Rice','Sugarcane','Wheat','Cotton'])
        irrigation = st.selectbox("Irrigation Type", ['Drip','Rainfed','Sprinkler','Canal'])

    # --- Prepare Input Data Dynamically ---
    user_input = {
        'Soil_Moisture': Soil_Moisture,
        'Temperature_C': Temperature_C,
        'Humidity': Humidity,
        'Rainfall_mm': Rainfall_mm,
        'Wind_Speed_kmh': Wind_Speed_kmh,
        'Previous_Irrigation_mm': Previous_Irrigation_mm,
        'Crop_Growth_Stage': Crop_Growth_Stage,
        'Mulching_Used': Mulching_Used,
        'Crop_Type_Maize': 1 if crop=='Maize' else 0,
        'Crop_Type_Potato': 1 if crop=='Potato' else 0,
        'Crop_Type_Rice': 1 if crop=='Rice' else 0,
        'Crop_Type_Sugarcane': 1 if crop=='Sugarcane' else 0,
        'Crop_Type_Wheat': 1 if crop=='Wheat' else 0,
        # Cotton dropped → implicitly 0
        'Irrigation_Type_Drip': 1 if irrigation=='Drip' else 0,
        'Irrigation_Type_Rainfed': 1 if irrigation=='Rainfed' else 0,
        'Irrigation_Type_Sprinkler': 1 if irrigation=='Sprinkler' else 0,
        # Canal dropped → implicitly 0
        'Water_Source_Rainwater': 1 if water=='Rainwater' else 0,
        'Water_Source_Reservoir': 1 if water=='Reservoir' else 0,
        'Water_Source_River': 1 if water=='River' else 0,
        # Groundwater dropped → implicitly 0
    }

    input_df = pd.DataFrame([user_input])
    # Fill missing features dynamically
    for col in pipeline.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[pipeline.feature_names_in_]

    # --- Prediction ---
    if st.button("💧 Predict Irrigation Need"):
        prediction = pipeline.predict(input_df)
        prediction_proba = pipeline.predict_proba(input_df)
        pred_class = prediction[0]

        label_map = {0:"Low",1:"Medium",2:"High"}
        description_map = {0:"Minimal irrigation required.",1:"Moderate irrigation required.",2:"High irrigation required."}
        colors = {0:"#2ecc71",1:"#f1c40f",2:"#e74c3c"}

        st.subheader("Prediction")
        st.markdown(f"""
            <div style='
                background-color:{colors[pred_class]};
                padding:15px;
                border-radius:8px;
                color:white;
                font-size:20px;
                font-weight:bold;
                text-align:center;
            '>
            Irrigation Need: {label_map[pred_class]}
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**Interpretation:** {description_map[pred_class]}")

        st.subheader("Prediction Probabilities")
        col1, col2, col3 = st.columns(3)
        col1.metric("Low", f"{prediction_proba[0][0]*100:.2f}%")
        col2.metric("Medium", f"{prediction_proba[0][1]*100:.2f}%")
        col3.metric("High", f"{prediction_proba[0][2]*100:.2f}%")

        st.subheader("Irrigation Intensity Level")
        st.progress(int(prediction_proba[0][2]*100))

        if pred_class == 2:
            st.error("⚠️ High Irrigation Required")
        elif pred_class == 1:
            st.warning("⚠️ Moderate Irrigation Required")
        else:
            st.success("✅ Low Irrigation Required")

# =========================
# --- REPORT TAB ---
# =========================
if menu == "Report":
    st.markdown("<h1 style='color:blue'> Predicting Irrigation Need Using Extreme Gradient Boosting.</h1>", unsafe_allow_html=True)

    # Sidebar for jumping to sections
    section = st.sidebar.radio(
        "Jump to Section",
        ["Introduction", "Problem Statement", "Data Description", "EDA",
         "Data Preprocessing", "Methodology", "Modeling & Evaluation",
         "Results & Insights", "Recommendations", "Conclusion"]
    )

    # --- Introduction ---
    if section == "Introduction":
        st.markdown("## Introduction")
        st.markdown("### Background")
        st.markdown(
            "Agriculture increasingly relies on data-driven solutions to improve productivity and "
            "sustainability. One critical aspect is efficient irrigation, as both over-irrigation "
            "and under-irrigation can negatively impact crop yield and resource utilization. With "
            "advancements in machine learning, it is now possible to predict irrigation needs using "
            "environmental and soil-related data."
        )
        st.markdown(
            'The "Predicting Irrigation Need" dataset is a synthetic dataset generated from a deep '
            "learning model trained on a real-world irrigation prediction dataset. While it closely "
            "reflects real agricultural conditions, it introduces slight variations, making it suitable "
            "for developing and testing robust predictive models."
        )
        st.markdown("### Relevance")
        st.markdown(
            "Efficient water management is a major challenge in modern agriculture, especially in "
            "regions facing water scarcity and climate variability. Smart irrigation systems are becoming "
            "increasingly important in precision agriculture, where decisions are guided by data rather than "
            "intuition."
        )
        st.markdown("**Predicting irrigation needs helps:**")
        st.markdown("- Optimize water usage and reduce wastage")
        st.markdown("- Improve crop health and yield")
        st.markdown("- Support climate-resilient farming practices")
        st.markdown("- Reduce operational costs for farmers")
        st.markdown("**Highly relevant across sectors:**")
        st.markdown("- Agritech and smart farming systems")
        st.markdown("- Environmental and water resource management")
        st.markdown("- Climate-smart agriculture initiatives")
        st.markdown("- Government and NGO agricultural programs")
        st.markdown("### Objective of the Analysis")
        st.markdown("- Develop a machine learning model that predicts irrigation needs")
        st.markdown("- Identify key factors influencing irrigation requirements")
        st.markdown("- Build and evaluate predictive models for decision-making")
        st.markdown("- Support intelligent irrigation systems for sustainable agriculture")

    # --- Problem Statement ---
    elif section == "Problem Statement":
        st.markdown("## Problem Statement")
        st.markdown("### Research Problem")
        st.markdown(
            "Agricultural productivity and sustainability are highly dependent on efficient water usage. "
            "However, farmers often rely on manual judgment or fixed irrigation schedules, which can"
            "lead to over-irrigation or under-irrigation. This results in water wastage, increased costs, "
            "and reduced crop yields. The core problem is the lack of a reliable, data-driven system" 
            "that can determine when irrigation is actually needed based on real-time environmental"
            "and soil conditions. Therefore, there is a need to develop a predictive solution that can"
            "support informed irrigation decisions and optimize water usage in agriculture. "
        )
        st.markdown("### Target Variable")
        st.markdown("**Predicted / Classified:**")
        st.markdown("The model predicts the level of irrigation required under given environmental and soil conditions. This is a multiclass classification problem with three categories: ")
        st.markdown("- Low irrigation need ")
        st.markdown("- High irrigation need ")
        st.markdown("- Medium irrigation need ")

        st.markdown("**Optimized:**")
        st.markdown("The goal is to optimize water usage by ensuring irrigation is applied at the appropriate level. This helps in: ")
        st.markdown("- Minimizing water wastage ")
        st.markdown("- Improving crop health and yield ")
        st.markdown("- Enhancing efficiency in irrigation systems ")


    # --- Data Description ---
    elif section == "Data Description":
        st.markdown("## Data Description")
        st.markdown("### Source of the Data")
        st.markdown(
            "The data is synthetically generated using a deep learning model trained on an original irrigation prediction dataset."
        )
        st.markdown("### Features / Variables")
        st.markdown(
            "The model uses 11 key features selected via RFE to predict irrigation needs, including "
            "environmental factors (soil moisture, temperature, humidity, rainfall, wind speed), "
            "crop-specific factors (crop growth stage, crop type), and management practices (mulching, previous irrigation). "
            "Irrigation type and water source are also included."
        )
        st.markdown("### Number of Records and Variables")
        st.markdown("The dataset contains 630000 observations and 20 variables.")
        st.markdown("### Missing Values")
        st.markdown("All variables have complete observations with 630000 non-null values per column.")
        st.markdown("### Data Types")
        st.markdown("11 numerical features (float64) and 9 categorical features (object).")

    # --- EDA ---
    elif section == "EDA":
        st.markdown("## Exploratory Data Analysis (EDA)")
        st.subheader("Target Class Distribution")
        st.image("Irrigation need count.png", width=700)
        st.markdown(
            "The dataset for predicting irrigation need is highly imbalanced, with 58.72% of" 
            "observations labeled as Low, 37.95% as Medium, and only 3.33% as High. This imbalance" 
            "means the model has many more examples of Low irrigation than High, which can bias"
            "predictions toward the majority class."
        )
        st.subheader("Irrigation Need Across Crop Growth Stages")
        st.image("Crop Growth Stage vs Irrigation Need.png", width=700)
        st.markdown(
            "This stacked bar chart shows how irrigation needs vary across different crop growth"
            "stages. The Flowering and Vegetative stages have the highest demand for water, with many"
            "cases requiring medium to high irrigation, since plants need more water during active"
            "growth and reproduction. In contrast, the Harvest and Sowing stages are mostly" 
            "associated with low irrigation needs, as crops either require less water or too much water can be harmful. "
        )
        st.subheader("Mulching Used vs Crop Growth Stages")
        st.image("Mulching Used vs Irrigation Need.png", width=700)
        st.markdown(
            "This stacked bar chart illustrates the impact of mulching on irrigation needs. Farms "
            "without mulching have a higher demand for water, with a significant number of crops"
            "requiring Medium (157,096) or High (18,521) irrigation, while Low irrigation accounts for"
            "140,836 instances. In contrast, farms using mulch show a dramatic shift toward Low"
            "irrigation (229,081), with far fewer crops needing Medium (81,978) or High (2,488) water."
            "This demonstrates that mulching effectively retains soil moisture, reducing overall" 
            "irrigation requirements and making it a key practice for water-efficient farming. "
        )

    # --- Data Preprocessing ---
    elif section == "Data Preprocessing":
        st.markdown("## Data Preprocessing")
        st.markdown("### Handling Missing Data")
        st.markdown(
                    "The dataset was first assessed for missing values and the result showed no" 
                    "missing values, indicating that the dataset was complete. Therefore, no imputation" 
                    "or removal of observations was required."
        )
        st.markdown("### Encoding of Variables")
        st.markdown("To prepare the data for machine learning modeling, different types of variables" 
                "were encoded appropriately: ")
    
        st.markdown("- Binary Variables Binary categorical variables were transformed into numerical"
                    "format by mapping")
        st.markdown("- Ordinal Variables Ordinal variables which have a natural order, were encoded" 
                    "using predefined rankings to preserve their inherent structure. ")
        st.markdown("- Nominal Variables Nominal variables (those without any inherent order) were" 
                    "encoded using one-hot encoding, creating binary indicator variables for each category.")

        st.markdown("### Feature Scaling")
        st.markdown(
                "Feature scaling was not applied in this study. This is because the final model used was"
                "XGBoost, a tree-based algorithm that is invariant to feature scaling. Tree-based models" 
                "split data based on feature thresholds rather than distances, making scaling unnecessary."
        )

    # --- Methodology ---
    elif section == "Methodology":
        st.markdown("## Methodology")
        st.markdown("### Models")
        st.markdown(
                "The XGBoost model was chosen because it attained the highest cross-validated F1 score"
                "(0.9700) among all the models tested. Before selecting XGBoost, several other models" 
                "were trained and evaluated, including: LightGBM, Random Forest, K-Nearest Neighbors"
                "(KNN), Decision Tree, Logistic Regression XGBoost, an advanced ensemble learning"
                "algorithm, builds multiple decision trees sequentially, with each tree correcting the errors"
                "of the previous ones. In this study, it predicts irrigation needs based on environmental, " 
                "soil, and crop conditions, offering highly accurate classification to support efficient water" 
                "use, improved crop health, and sustainable farming practices. "
        )
        st.markdown("### Train-Test Split")
        st.markdown(
                "To evaluate model performance, the dataset was split into training and testing sets. 80% of"
                "the data was used for training the models, while 20% was reserved for testing to assess"
                "how well the model generalizes to unseen data. The split was stratified to ensure that the"
                "proportion of classes (irrigation needed vs. not needed) was maintained in both sets, "
                "preserving the balance of the target variable. This approach allows for reliable evaluation"
                "of model performance and helps prevent overfitting, ensuring that predictions remain"
                "accurate when applied in real-world irrigation scenarios. "
        )
        st.markdown("### Feature Engineering")
        st.markdown(
                "Feature engineering was performed to prepare the dataset for machine learning and  " 
                "enhance model performance. Categorical variables were encoded into numerical  " 
                "formats: binary variables as 0/1, ordinal variables mapped to ordered numbers, and " 
                " nominal variables one-hot encoded. Continuous variables were retained as-is to " 
                "preserve precise environmental and soil information. To reduce complexity and focus on" 
                "the most important predictors, Recursive Feature Elimination (RFE) with XGBoost was " 
                "applied, selecting the 11 most relevant features while maintaining predictive power. "
        )

    # --- Modeling & Evaluation ---
    elif section == "Modeling & Evaluation":
        st.markdown("## Modeling & Evaluation")
        st.subheader("Classification Report")
        st.image("classification report.png", width=700)
        st.markdown(
                "The model performed exceptionally well in predicting irrigation needs, achieving an overall" 
                "accuracy of 0.98 on the test set of 126,000 observations. For individual classes, precision"
                "ranged from 0.96 to 0.99, indicating that most predictions were correct, while recall "
                "ranged from 0.91 to 0.99, showing that the model successfully captured the majority of" 
                "actual cases in each class. The F1-scores, which balance precision and recall, were also" 
                "high (0.94-0.99), reflecting strong predictive performance across low, medium, and high " 
                "irrigation levels. "
    )
        st.subheader("Confusion Report")
        st.image("confusion matrix.png", width=700)
        st.markdown(
                "The model demonstrates very strong performance in predicting Low irrigation (Class 0), " 
                "correctly classifying 73,587 instances, with only 396 misclassified as Medium and none as  High."
    )

        st.markdown(
                "For Medium irrigation (Class 1), the model correctly predicts 46,619 cases, although some"  
                "misclassification occurs, with 1,049 instances predicted as Low and 147 as High, " 
                "suggesting moderate overlap with neighboring classes. For High irrigation (Class 2), the" 
                "model correctly identifies 3,830 instances but misclassifies 372 cases as Medium, " 
                "indicating some difficulty in distinguishing between Medium and High irrigation levels. The"
                "model performs very well, with minor challenges in differentiating between adjacent" 
                "categories."
    )

    # --- Results & Insights ---
    elif section == "Results & Insights":
        st.markdown("## Results & Insights")
        st.markdown(
                "The analysis revealed that XGBoost was the best-performing model, achieving the highest" 
                "cross-validated F1 score, indicating strong predictive accuracy for irrigation needs. Key" 
                "features influencing irrigation requirements included Crop Growth Stage, Mulching, and" 
                "Soil Moisture. Patterns observed showed that low soil moisture combined with high" 
                "temperature and advanced"
        )

    # --- Recommendations ---
    elif section == "Recommendations":
        st.markdown("## Recommendations")
        st.markdown("1. **Prioritize Key Factors in Irrigation Planning:** \nFocus on monitoring Crop Growth Stage, Mulching practices, and Soil Moisture, as these features have the highest influence on irrigation needs.")
        st.markdown("2. **Implement Data-Driven Irrigation Scheduling:** \nUse predictive models like XGBoost to determine when irrigation is necessary, reducing water wastage and improving crop yield.")
        st.markdown("3. **Adopt Smart Irrigation Technologies:** \nConsider drip or sprinkler systems integrated with environmental sensors to optimize water use efficiently.")
        st.markdown("4. **Train Farmers and Stakeholders:** \nEducate farmers on the importance of mulching, soil moisture monitoring, and growth stage assessment to support smarter water management.")
        st.markdown("5. **Integrate with Climate-Smart Practices:** \nIncorporate predictions into broader sustainable farming initiatives, ensuring resilience against climate variability and water scarcity.")
    
    # --- Conclusion ---
    elif section == "Conclusion":
        st.markdown("## Conclusion")
        st.markdown(
                "The project aimed to predict irrigation needs to optimize water usage in agriculture. Using " 
                "multiple machine learning models, XGBoost was chosen for its superior performance, "
                "with key factors like Crop Growth Stage, Mulching, and Soil Moisture driving "
                "predictions. The analysis highlights that data-driven irrigation decisions can "
                "reduce water waste, improve crop yield, and support sustainable farming practices."
                )

# --- Footer ---
st.markdown("---")
st.caption("Irrigation Prediction Model | Machine Learning Deployment 🌱")