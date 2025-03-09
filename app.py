import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

# 🎯 Page Title & Description
st.title("💪🏻 Personal Fitness Tracker")
st.write("### Track your calorie burn and fitness progress effortlessly! 🚀")
st.write("Pass your parameters such as `Age`, `Gender`, `Height`, `Weight`, etc., and get your predicted calorie burn.")
st.markdown("---")

# 📌 Sidebar for User Input
st.sidebar.header("📌 User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("🎂 Age", 10, 100, 30)
    height = st.sidebar.slider("📏 Height (cm)", 120, 220, 170)
    weight = st.sidebar.slider("⚖️ Weight (kg)", 30, 150, 70)
    duration = st.sidebar.slider("⏳ Duration (min)", 0, 120, 15)
    heart_rate = st.sidebar.slider("❤️ Heart Rate (bpm)", 60, 130, 80)
    body_temp_f = st.sidebar.slider("🌡️ Body Temperature (°F)", 96, 108, 98)

    # Convert Fahrenheit to Celsius
    body_temp_c = round((body_temp_f - 32) * 5/9, 2)

    gender_button = st.sidebar.radio("👫 Gender", ("Male", "Female"))
    gender = 1 if gender_button == "Male" else 0

    data_model = {
        "Age": age,
        "Height": height,
        "Weight": weight,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp_c,
        "Gender_male": gender
    }

    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

# 🎯 Display User Input
st.header("🎯 Your Parameters")
st.dataframe(df, use_container_width=True)

# 📂 Loading and Preprocessing Data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = round(data["Weight"] / ((data["Height"] / 100) ** 2), 2)

# ✅ Selecting relevant columns (Height & Weight instead of BMI)
exercise_train_data = exercise_train_data[["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]

exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

# Splitting Features & Labels
X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]
X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# 🏋️ Train Model
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

df = df.reindex(columns=X_train.columns, fill_value=0)

# 🔮 Predict Calories
prediction = random_reg.predict(df)

st.markdown("---")
st.header("🔥 Predicted Calories Burned")
st.success(f"{round(prediction[0], 2)} kcal 🔥")

# 🔍 Show Similar Results
st.header("🔍 Similar Caloric Burn Cases")
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
st.dataframe(similar_data.sample(5) if len(similar_data) > 5 else similar_data, use_container_width=True)
st.markdown("---")

# 📊 Insights & Comparisons
st.header("📊 General Information")
boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

st.write(f"📌 You are older than **{round(sum(boolean_age) / len(boolean_age) * 100, 2)}%** of other users.")
st.write(f"📌 Your exercise duration is longer than **{round(sum(boolean_duration) / len(boolean_duration) * 100, 2)}%** of others.")
st.write(f"📌 Your heart rate is higher than **{round(sum(boolean_heart_rate) / len(boolean_heart_rate) * 100, 2)}%** of other users during exercise.")
st.write(f"📌 Your body temperature is higher than **{round(sum(boolean_body_temp) / len(boolean_body_temp) * 100, 2)}%** of other users during exercise.")
