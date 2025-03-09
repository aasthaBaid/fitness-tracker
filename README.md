# Fitness Tracker

## Overview
The **Fitness Tracker** web application helps users estimate the number of calories burned based on their personal details and exercise data. Built with **Streamlit** and powered by a **Random Forest Regressor**, this tool provides quick, data-driven insights into fitness progress without requiring an account.

## Features
- ✨ **Instant Calorie Prediction**: Input your age, gender, BMI, workout duration, heart rate, and body temperature to get an estimated calorie burn.
- 📈 **Insightful Comparisons**: See how your stats compare to other users in terms of age, heart rate, and workout duration.
- 👀 **Similar Results Lookup**: Get examples of people with similar calorie burn metrics.
- 🔄 **User-Friendly**: No need to create an account—just enter your details and get results instantly!

## How It Works
1. **Enter your details** in the sidebar, including your age, BMI, exercise duration, heart rate, and body temperature.
2. **The model predicts** your calorie burn using a trained **Random Forest** algorithm.
3. **Compare your stats** with others and get suggestions for improvement.
4. **Use the insights** to optimize your workout and track progress over time!

## Technologies Used
- **Python**
- **Streamlit**
- **Pandas**
- **Scikit-learn**
- **Random Forest Regressor**

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fitness-tracker.git
   cd fitness-tracker
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Contributing
We welcome contributions! Feel free to fork this repository and submit a pull request with improvements.

## License
This project is open-source and available under the MIT License.

