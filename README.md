# 🛠️ Predictive Maintenance for HVAC Pumps Using Pump Sensor Data

## 📌 Objective
This project aims to **predict failures in HVAC pumps** using pump sensor data. Predictive maintenance reduces downtime, prevents costly repairs, and improves energy efficiency by identifying early signs of failure from sensor behavior.

---

## 🧰 Tools Used
- Python
- pandas
- scikit-learn
- XGBoost
- Streamlit
- FastAPI

---

## 📊 1. Data Exploration & Preprocessing

**Dataset Overview**
- ~220,000 rows × 50+ sensor columns
- Target variable: `machine_status` (`NORMAL`, `RECOVERING`, `BROKEN`)

**Missing Values Handling**
- Dropped `sensor_15` (100% missing)
- Dropped `sensor_50` (35%+ missing)
- Remaining missing values: Filled using **median imputation**

**Visualizations**
- Distribution of machine status (bar chart)
- Outlier detection using boxplots

**Time Features**
- Extracted `hour` and `dayofweek` from timestamps for temporal insights

---

## ⚙️ 2. Feature Engineering

| Feature       | Type         | Description                                 |
|---------------|--------------|---------------------------------------------|
| `rolling_mean`| Local average | Mean over 5-minute window                   |
| `rolling_std` | Local variability | Captures fault signal variance         |
| `diff`        | Change rate   | Instant change between sensor readings      |
| `lag_1`       | Lag           | Previous value to capture trends            |
| `hour`, `dayofweek` | Time features | Time-of-day influence on failures  |

- **Scaling**: All features scaled using `StandardScaler`
- **Selection**: Used `SelectKBest` with ANOVA F-test for feature selection

---

## 🤖 3. Model Development

### ✅ Final Model: **Random Forest Classifier**

**Training Details**
- Train-Test Split: 80/20 (time-aware)
- Scaled features with `StandardScaler`

**Evaluation Metrics (XGBoost shown for comparison):**

| Metric     | Random Forest | XGBoost | Logistic Regression | SVM   |
|------------|----------------|---------|----------------------|-------|
| F1 Score   | 0.99           | 0.92    | 0.90                 | 0.91  |
| Precision  | 0.89           | 0.87    | 0.83                 | 0.84  |
| Recall     | 0.98           | 0.97    | 0.98                 | 0.99  |

---

## 💾 4. Saving the Model

Saved the model and preprocessing objects using `joblib`:

- `best_model.pkl` (Random Forest)
- `scaler.pkl` (StandardScaler)
- `selector.pkl` (Feature selector)

---

## 🔍 5. Recommendations & Insights

**Key Observations**
- Failures often preceded by **spikes in `rolling_std`** of critical sensors
- **Time-of-day** (especially early hours) linked to higher failure rates
- Top influential sensors: `sensor_07`, `sensor_14`, `sensor_48`

---

## ⚡ 6. FastAPI Deployment

FastAPI is used to deploy a lightweight REST API that provides real-time predictions based on sensor input.

### 🚀 Step-by-Step Execution

1. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn joblib scikit-learn xgboost
