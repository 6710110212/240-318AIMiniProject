import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Dataset
df = pd.read_csv(
    "Sleep_Health_and_Lifestyle_Dataset.csv"
)

# Load Trained Model
best_model = joblib.load(
    "sleep_quality_model_fn.pkl"
)
best_name = "Random Forest"

# Page Configuration
st.set_page_config(
    page_title="Sleep Quality Prediction",
    page_icon="🛌",
    layout="centered"
)

# Custom CSS
st.markdown(
    """
    <style>

    .main {
        background-color: #f5f7f8;
    }

    .block-container {
        max-width: 620px;
        padding-top: 35px;
        padding-bottom: 40px;
    }

    div.stButton > button {
        display: block;
        width: 250px;
        height: 48px;
        margin: 20px auto 10px auto;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 500;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Dashboard Title
st.html(
    """
    <div style="
        width:600px;
        max-width:100%;
        margin:0 auto 20px auto;
        text-align:center;
    ">

        <h1 style="
            color:#2c3e50;
            margin:0;
            font-size:28px;
        ">
            🛌 Sleep Quality Prediction
        </h1>

        <p style="
            color:#7f8c8d;
            font-size:14px;
            margin-top:8px;
        ">
            Enter your daily lifestyle information
            to predict your quality of sleep.
        </p>

    </div>
    """
)

# Personal Information
st.html(
    """
    <div style="
        width:600px;
        max-width:100%;
        margin:10px auto 10px auto;
        padding-bottom:6px;
        border-bottom:2px solid #e5e7e9;
    ">

        <h3 style="
            color:#2c3e50;
            margin:0;
            font-size:18px;
        ">
            👤 Personal Information
        </h3>

    </div>
    """
)


# Age
age = st.slider(
    "Age",
    min_value=int(df["Age"].min()),
    max_value=int(df["Age"].max()),
    value=25,
    step=1
)


# Gender
gender = st.selectbox(
    "Gender",
    sorted(
        df["Gender"]
        .dropna()
        .unique()
    )
)

# Occupation
occupation = st.selectbox(
    "Occupation",
    sorted(
        df["Occupation"]
        .dropna()
        .unique()
    )
)

# BMI Category
bmi_category = st.selectbox(
    "BMI Category",
    sorted(
        df["BMI Category"]
        .dropna()
        .unique()
    )
)

# Daily Lifestyle & Health
st.html(
    """
    <div style="
        width:600px;
        max-width:100%;
        margin:20px auto 10px auto;
        padding-bottom:6px;
        border-bottom:2px solid #e5e7e9;
    ">

        <h3 style="
            color:#2c3e50;
            margin:0;
            font-size:18px;
        ">
            ❤️ Daily Lifestyle & Health
        </h3>

    </div>
    """
)

# Sleep Duration
sleep_duration = st.slider(
    "Sleep Duration",
    min_value=float(
        df["Sleep Duration"].min()
    ),
    max_value=float(
        df["Sleep Duration"].max()
    ),
    value=7.5,
    step=0.1
)

# Physical Activity
physical_activity = st.slider(
    "Physical Activity",
    min_value=int(
        df["Physical Activity Level"].min()
    ),
    max_value=int(
        df["Physical Activity Level"].max()
    ),
    value=60,
    step=1
)

# Stress Level
stress_level = st.slider(
    "Stress Level",
    min_value=int(
        df["Stress Level"].min()
    ),
    max_value=int(
        df["Stress Level"].max()
    ),
    value=4,
    step=1
)

# Heart Rate
heart_rate = st.slider(
    "Heart Rate",
    min_value=int(
        df["Heart Rate"].min()
    ),
    max_value=int(
        df["Heart Rate"].max()
    ),
    value=70,
    step=1
)

# Daily Steps
daily_steps = st.slider(
    "Daily Steps",
    min_value=int(
        df["Daily Steps"].min()
    ),
    max_value=int(
        df["Daily Steps"].max()
    ),
    value=8000,
    step=100
)

# Predict Button
predict_button = st.button(
    "✓  Predict",
    type="primary"
)

# Prediction
if predict_button:

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Occupation": [occupation],
        "Sleep Duration": [sleep_duration],
        "Physical Activity Level": [physical_activity],
        "Stress Level": [stress_level],
        "BMI Category": [bmi_category],
        "Heart Rate": [heart_rate],
        "Daily Steps": [daily_steps]
    })


    # Predict using trained Pipeline

    prediction = best_model.predict(
        input_data
    )[0]


    # Keep prediction within expected 0-10 range

    prediction_display = np.clip(
        prediction,
        0,
        10
    )

    # Determine status
    if prediction_display >= 8:

        status = "High"
        status_icon = "🟢"

    elif prediction_display >= 6:

        status = "Moderate"
        status_icon = "🟡"

    else:

        status = "Low"
        status_icon = "🔴"

    # Result Card
    st.html(
        f"""
        <div style="
            width:560px;
            max-width:100%;
            box-sizing:border-box;
            margin:20px auto 0 auto;
            padding:28px 25px;
            border-radius:16px;
            background:#ffffff;
            border:1px solid #e5e7e9;
            box-shadow:0 4px 15px rgba(0,0,0,0.08);
            text-align:center;
        ">

            <div style="
                color:#7f8c8d;
                font-size:14px;
                margin-bottom:8px;
            ">
                PREDICTION RESULT
            </div>


            <h2 style="
                color:#2c3e50;
                margin:0 0 15px 0;
                font-size:22px;
            ">
                Predicted Quality of Sleep
            </h2>


            <div style="
                font-size:52px;
                font-weight:bold;
                color:#2c3e50;
                line-height:1.1;
                margin:10px 0;
            ">
                {prediction_display:.2f}

                <span style="
                    font-size:24px;
                    color:#7f8c8d;
                    font-weight:normal;
                ">
                    / 10
                </span>

            </div>


            <div style="
                margin-top:15px;
                padding:10px;
                border-radius:10px;
                background:#f8f9f9;
                font-size:18px;
            ">

                {status_icon}
                Status:
                <b>{status}</b>

            </div>


            <div style="
                margin-top:18px;
                padding-top:12px;
                border-top:1px solid #eeeeee;
                color:#7f8c8d;
                font-size:13px;
            ">

                Model used:
                <b>{best_name}</b>

            </div>

        </div>
        """
    )