Project Overview

The Car Price Prediction project focuses on building a machine learning model to estimate the selling price of used cars based on various features. The objective of this project is to analyze how different car attributes influence resale value and to develop a regression model capable of making accurate predictions.

This project demonstrates practical implementation of data preprocessing, exploratory data analysis (EDA), regression modeling, and performance evaluation using Python.

🎯 Objective

To build a predictive model that can accurately estimate the selling price of a car using its features such as:

Year of manufacture

Present price

Kilometers driven

Fuel type

Seller type

Transmission type

Ownership details

🧠 Approach
1️⃣ Data Preprocessing

Removed irrelevant columns such as car name

Handled categorical variables using one-hot encoding

Ensured all features were in numerical format

Split the dataset into training (80%) and testing (20%) sets

Data preprocessing was essential to make the dataset suitable for machine learning algorithms.

2️⃣ Model Development

A Linear Regression model was used to predict car selling prices.

Linear Regression was selected because:

It is suitable for regression problems

It provides interpretable coefficients

It performs well when relationships between variables are approximately linear

The model was trained using the training dataset and evaluated on unseen test data.

3️⃣ Model Evaluation

The model performance was measured using:

Mean Absolute Error (MAE) – measures average prediction error

Mean Squared Error (MSE) – penalizes larger errors

R² Score – indicates how well the model explains variance

The model achieved a strong R² score, indicating good predictive performance.

📊 Visualizations

To better understand the dataset and model behavior, multiple visualizations were created:

📈 Correlation Heatmap – to analyze relationships between features

📊 Price Distribution Plot – to study data spread

📉 Actual vs Predicted Plot – to evaluate prediction accuracy

📌 Feature Importance Graph – to identify key influencing features

📉 Residual Plot – to analyze prediction errors

All visualizations are automatically saved in the images folder after running the script.

📂 Project Structure
Car Price Prediction/
│
├── data/
│   └── car_data.csv
│
├── images/
│
├── notebooks/
│   └── car_price_prediction.py
│
└── README.md
▶️ How to Run the Project

Open the project folder in VS Code

Open the terminal

Navigate to the notebooks folder:

cd notebooks

Run the script:

python car_price_prediction.py

All output graphs will be saved inside the images folder.

💡 Key Insights

Present price has a strong impact on selling price

Transmission and fuel type influence resale value

Older vehicles generally have lower selling prices

The regression model performs well with minimal prediction error

🚀 Conclusion

This project showcases the complete machine learning workflow, from data preprocessing to model evaluation and visualization. It demonstrates practical skills in regression modeling, feature analysis, and data interpretation using Python.
