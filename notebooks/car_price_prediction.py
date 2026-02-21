# ==========================================
# CAR PRICE PREDICTION PROJECT
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# SET PATHS (Professional Structure)
# ==========================================

base_path = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(base_path, "data", "car_data.csv")
images_path = os.path.join(base_path, "images")

os.makedirs(images_path, exist_ok=True)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(data_path)
print("✅ Dataset Loaded Successfully!")

# ==========================================
# DATA PREPROCESSING
# ==========================================

# Drop Car Name column (not useful for regression)
if "Car_Name" in df.columns:
    df.drop("Car_Name", axis=1, inplace=True)

# Convert categorical columns into numbers
df = pd.get_dummies(df, drop_first=True)
print("✅ Encoding Completed!")

# ==========================================
# SPLIT DATA
# ==========================================

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# TRAIN MODEL
# ==========================================

model = LinearRegression()
model.fit(X_train, y_train)

print("✅ Model Training Completed!")

# ==========================================
# MODEL EVALUATION
# ==========================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print(f"MAE: {round(mae,2)}")
print(f"MSE: {round(mse,2)}")
print(f"R2 Score: {round(r2,2)}")

# ==========================================
# VISUALIZATIONS
# ==========================================

# 1️⃣ Correlation Heatmap
plt.figure(figsize=(10,8))
corr = df.corr()
plt.imshow(corr)
plt.colorbar()
plt.title("Correlation Heatmap")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.tight_layout()
plt.savefig(os.path.join(images_path, "correlation_heatmap.png"))
plt.close()

# 2️⃣ Price Distribution
plt.figure()
plt.hist(y, bins=20)
plt.title("Car Price Distribution")
plt.xlabel("Selling Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(images_path, "price_distribution.png"))
plt.close()

# 3️⃣ Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred)

min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val])

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.tight_layout()
plt.savefig(os.path.join(images_path, "actual_vs_predicted.png"))
plt.close()

# 4️⃣ Feature Importance
importance = pd.Series(model.coef_, index=X.columns)
importance = importance.sort_values(ascending=False)

plt.figure(figsize=(10,6))
importance.head(10).plot(kind="bar")
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.savefig(os.path.join(images_path, "feature_importance.png"))
plt.close()

# 5️⃣ Residual Plot
residuals = y_test - y_pred

plt.figure()
plt.scatter(y_pred, residuals)
plt.axhline(y=0)
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig(os.path.join(images_path, "residual_plot.png"))
plt.close()

print("\n🎉 Project Completed Successfully!")
print("📊 All graphs saved inside images folder.")