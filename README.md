# 🖥️ Churn Prediction & Segmentation For Retention Strategy For Ecommerce | Machine Learning - Python

---

<img width="1536" height="1024" alt="ChatGPT Image 16_06_37 18 thg 7, 2025" src="https://github.com/user-attachments/assets/4b73d69c-c5c2-41d5-82a0-740adf7df6cc" />

Author: Duong Chi Tuan  
Date: July 2025  
Tools Used: Python   

---

## 📑 Table of Contents  

1. [📌 Background & Overview](#-background--overview)  
2. [📂 Dataset Description & Data Structure](#-dataset-description--data-structure)   
3. [📊 Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)  
4. [🧮 Train & Apply Churn Prediction Model](#-train-apply-churn-prediction)  

---

## 📌 Background & Overview  

### 🎯 Objective:

The goal is to build a system that analyzes and **predicts user churn** based on customer **behavior** and **usage characteristics**, enabling the company to:

- Identify early those users at high risk of churn, so that the business can proactively engage and retain customers.  

- Analyze the distinctive behavioral patterns of churned users to understand the root causes driving churn.  

- Segment churned users into actionable groups, allowing for personalized promotions or targeted retention policies tailored to each segment.   

### 👤 Who is this project for?  

- Data Analysts & Business Analysts
    
- Marketing & Customer Retention Teams  
---

## 📂 Dataset Description & Data Structure  

### 📌 Data Source  
- Source: The dataset is obtained from the e-commerce company's database.
- Size: The dataset contains 5,630 rows and 20 columns.
- Format: .xlxs file format.
### 📊 Data Structure & Relationships  

#### 1️⃣ Tables Used:  
The dataset contains only 1 table with customer and transaction-related data.  
#### 2️⃣ Table Schema & Data Snapshot  
  
<details>
  <summary>Click to expand the table schema</summary>

| **Column Name**              | **Data Type** | **Description**                                              |
|------------------------------|---------------|--------------------------------------------------------------|
| CustomerID                   | INT           | Unique identifier for each customer                          |
| Churn                        | INT           | Churn flag (1 if customer churned, 0 if active)              |
| Tenure                       | FLOAT         | Duration of customer's relationship with the company (months)|
| PreferredLoginDevice         | OBJECT        | Device used for login (e.g., Mobile, Desktop)                 |
| CityTier                     | INT           | City tier (1: Tier 1, 2: Tier 2, 3: Tier 3)                   |
| WarehouseToHome              | FLOAT         | Distance between warehouse and customer's home (km)         |
| PreferredPaymentMode         | OBJECT        | Payment method preferred by customer (e.g., Credit Card)     |
| Gender                       | OBJECT        | Gender of the customer (e.g., Male, Female)                  |
| HourSpendOnApp               | FLOAT         | Hours spent on app or website in the past month              |
| NumberOfDeviceRegistered     | INT           | Number of devices registered under the customer's account   |
| PreferedOrderCat             | OBJECT        | Preferred order category for the customer (e.g., Electronics)|
| SatisfactionScore            | INT           | Satisfaction rating given by the customer                    |
| MaritalStatus                | OBJECT        | Marital status of the customer (e.g., Single, Married)       |
| NumberOfAddress              | INT           | Number of addresses registered by the customer               |
| Complain                     | INT           | Indicator if the customer made a complaint (1 = Yes)         |
| OrderAmountHikeFromLastYear  | FLOAT         | Percentage increase in order amount compared to last year   |
| CouponUsed                   | FLOAT         | Number of coupons used by the customer last month            |
| OrderCount                   | FLOAT         | Number of orders placed by the customer last month           |
| DaySinceLastOrder            | FLOAT         | Days since the last order was placed by the customer        |
| CashbackAmount               | FLOAT         | Average cashback received by the customer in the past month  |

</details>   

---

## 📊 Exploratory Data Analysis (EDA)

### 1️⃣ Initial Exploration
[In 1]:  

```python

# Check the general information of df
dt.info()
```
[Out 1]:  

<img width="462" height="468" alt="image" src="https://github.com/user-attachments/assets/8b45cc6e-8cae-410b-bac7-5ced8b67a658" />
  
[In 2]:  

```python

# Check data summary
dt.head()
```
[Out 2]:  

<img width="1797" height="234" alt="image" src="https://github.com/user-attachments/assets/ddc00c83-e4fd-4aaa-bc42-f098047a885a" />
 
To understand the data structure and assess its quality, the following initial steps were taken:

- **Dataset Overview:**
  - The dataset contains **5,630 rows** and **20 columns**.
  - Used `.info()` to check data types and null values.
  - Used `.head()` to view sample transactions.

- **Data Type Review:**
  - No mismatched data types were detected.
  - Standardize the values in the column PreferredPaymentMode to ensure consistency.

- **Missing & Duplicate Values:**
  - `Tenure`: ~4.67% missing. 
  - `WarehouseToHome`: ~4.46% missing.
  - `HourSpendOnApp`: ~4.5% missing.
  - `OrderAmountHikeFromlastYear`: ~4.7% missing.
  - `CouponUsed`: ~4.5% missing.
  - `OrderCount`: ~4.6% missing.
  - `DaySinceLastOrder`: ~5.5% missing.
  - Duplicate rows: No duplicate values found.

### 2️⃣ Data Cleaning

The following steps were performed to clean the dataset and prepare it for segmentation:  

**Step 1:** Drop rows with missing values  

**Step 2:** Replace values with similar meanings in the PreferredPaymentMode column.  

## 🧮 Train & Apply Churn Prediction Model  

### 📝 Encoding

After preprocessing the dataset, encoding was applied to the categorical features:  

**1. One-Hot Encoding**  

- Categorical columns with a limited number of unique values were transformed using one-hot encoding, which creates separate True/False (Boolean) columns for each category. This format allows machine learning models to interpret categorical data effectively.

- The columns encoded are: PreferredLoginDevice, PreferredPaymentMode, PreferedOrderCat, MaritalStatus, Gender.

```python
# One-hot encoding
cat_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']
dt_encoded = pd.get_dummies(dt, columns=cat_cols, drop_first=True)
dt_encoded.head()
```

**2. Dropped Unnecessary Column**  

- The CustomerID column was removed, as it serves only as a unique identifier and does not carry predictive value for the model.

```python
dt = dt.drop(columns='CustomerID')
```

### 📝 Split Data into Features (X) and Target (y)  

After preprocessing the dataset, the next step was to separate the information into two main parts:  

- **Input variables (x):** these include all customer-related information such as behaviors and preferences, which are used to help the model learn.
- **Target variable (y):** this represents whether a customer has churned or not — the outcome we want the model to predict.

This separation ensures that the model focuses only on learning from relevant input data, and its performance can be evaluated based on how well it predicts the defined outcome.  

```python
# Split the data into features (x) and target (y)
x = dt_encoded.drop('Churn', axis=1)
y = dt_encoded['Churn']  # Target

# Split into training and testing sets (70/30 split)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
```  

### 📝 Standardize the Features Using MinMaxScaler  

All input features were normalized to a common range between **0 and 1** to ensure that no single feature dominates the model due to differences in scale.

The normalization process was applied as follows:

- Scaling parameters (minimum and maximum values) were calculated based only on the **training set**.

- Both the training and testing sets were then transformed using these parameters.

This approach prevents **data leakage**, ensuring that the model only learns from information available during training.  

```python
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
```

### 📝 Apply Model - Random Forest Classifier

- The Random Forest model was used to predict customer churn. This is a powerful machine learning algorithm that combines multiple decision trees to improve accuracy and stability.

- It delivered strong performance and helped identify key factors that influence churn, supporting more informed business decisions.

```python
# Initialize the Random Forest model with a fixed random state for reproducibility
model = RandomForestClassifier(random_state=42)

# Train the model using the scaled training data
model.fit(x_train_scaled, y_train)

# Make predictions on the scaled test data
y_pred = model.predict(x_test_scaled)

# Print the accuracy score of the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print a detailed classification report (precision, recall, F1-score, support)
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))  

# Import libraries for displaying the confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Visualize the confusion matrix with custom labels and color map
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Stay", "Churn"], cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

```
<img width="589" height="663" alt="image" src="https://github.com/user-attachments/assets/55538e5b-ab7b-43b0-af6b-7435d26e7318" />  

#### 🔎 Observation 

The initial Random Forest model achieved an overall accuracy of 94.3%, with a churn recall of 74.2%. While the model correctly identified most churners, it still missed about 1 in 4 of them — which could represent a significant loss for the business.  

#### 🔧 Hyperparameter Tuning to Improve Recall
To improve the model's ability to detect churned customers, we applied GridSearchCV using recall as the scoring metric.
This ensures that the model prioritizes correctly identifying churners, even at the cost of slightly lower precision or accuracy.  

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Initialize the baseline Random Forest model
clf_rand = RandomForestClassifier(random_state=42, class_weight='balanced')

# Define the hyperparameter grid for GridSearchCV to explore
param_grid = {
    'n_estimators': [50, 100, 200],        # Number of trees in the forest
    'max_depth': [None, 10, 20],           # Maximum depth of each tree
    'min_samples_split': [2, 5],           # Minimum samples required to split a node
    'min_samples_leaf': [1, 2],            # Minimum samples required at each leaf node
    'bootstrap': [True]                    # Whether to use bootstrap samples
}

# Use 'recall' as the scoring metric to prioritize correctly identifying churned users
grid_search = GridSearchCV(clf_rand, param_grid, cv=5, scoring='recall', n_jobs=-1, verbose=1)

# Fit the grid search on the scaled training data
grid_search.fit(x_train_scaled, y_train)

# Print out the best hyperparameter combination found
print("Best Parameters:", grid_search.best_params_)

# Make predictions on the test set and evaluate performance
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(x_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

```
🗝️ **Result:**  

The Random Forest model, tuned using GridSearchCV with **recall** as the scoring metric, achieved **92.9%** **accuracy** and **73.5%** **recall** for churned users. With a **precision** of **84.7%**, the model strikes a good balance between identifying churn and maintaining accuracy. This can be considered the final model for the churn prediction task.  

<img width="972" height="235" alt="image" src="https://github.com/user-attachments/assets/f9bbcfcd-d2c6-4e77-8ce8-ae6b6dd5ab45" />  

### 📝 Key Features Influencing User Churn  

```python
feature_importance = pd.Series(best_clf.feature_importances_, index=x.columns)
top_features = feature_importance.sort_values(ascending=False).head(10)

plt.figure(figsize=(8,5))
top_features.plot(kind='barh')
plt.gca().invert_yaxis()
plt.title("Top 10 Features That Affect Churn")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
```  

<img width="803" height="495" alt="image" src="https://github.com/user-attachments/assets/9d381944-a552-4119-b29b-31f3cf6b2405" />

```python
# Select columns to analyze
features_to_check = ['Tenure', 'OrderCount', 'CashbackAmount', 'Complain', 'WarehouseToHome', 'DaySinceLastOrder']

# Compare median values between Churned and Non-Churned users
comparison = dt_encoded.groupby('Churn')[features_to_check].median().T
comparison.columns = ['Not Churned (0)', 'Churned (1)']
comparison['Difference'] = comparison['Churned (1)'] - comparison['Not Churned (0)']

# Display the result
print("Median comparison between Churned and Non-Churned users:")
display(comparison.sort_values('Difference', ascending=False))
```

<img width="473" height="248" alt="image" src="https://github.com/user-attachments/assets/5a0915ce-1efb-419d-bcb5-4e4398494ef9" />  


📌 **Observation**

Based on both feature importance and behavioral median comparison, the following key differences were identified between **churned** and **non-churned** users:

| **Feature**                 | **Churned Group**        | **Non-Churned Group**     | **Insight**                                                                 |
|----------------------------|---------------------------|----------------------------|------------------------------------------------------------------------------|
| **Tenure**                 | 1 month                   | 10 months                 | New users tend to churn early, indicating weak onboarding or first impression. |
| **Complain**               | 1.0                       | 0.0                        | Users who submitted complaints are significantly more likely to churn.      |
| **CashbackAmount**         | ~149K                     | ~166K                      | Churned users received lower cashback incentives.                           |
| **DaySinceLastOrder**      | 2.5 days                  | 3 days                     | Churn tends to happen shortly after their most recent purchase.             |
| **SatisfactionScore**      | Lower                     | Higher                     | Poor satisfaction correlates with increased churn risk.                     |
| **OrderAmountHike (YoY)**  | Low growth                | Consistent growth          | Churned customers show weak engagement and lower spending momentum.         |  

🎯 **Recommendations**  

To mitigate churn and retain more users, the following actions are recommended:

1. **Strengthen Onboarding Programs**  
   - Implement a 30–90 day lifecycle campaign for new users with targeted assistance, educational content, and personalized offers.

2. **Proactive Complaint Management**  
   - Establish a dedicated complaint response system with SLAs under 24 hours. Offer loyalty points or discounts for unresolved issues.

3. **Revamp Incentive Strategy**  
   - Deliver more personalized cashback and coupon schemes, especially to low-engagement segments and first-time buyers.

4. **Monitor Post-Purchase Experience**  
   - Introduce NPS or satisfaction surveys within 1–3 days after delivery to detect and resolve dissatisfaction early.

5. **Address Fulfillment Gaps**  
   - Improve shipping transparency for users far from warehouses and offer expedited delivery options where possible.
  
### 📝 Customer Segmentation Using Clustering  

**Step 1: Feature Selection**  
Selected key behavioral features likely associated with churn: `Tenure`, `OrderCount`, `CashbackAmount`, `Complain`, `WarehouseToHome`, `DaySinceLastOrder`

**Step 2: Feature Scaling**  
Normalized the selected features using **StandardScaler** to ensure equal contribution across variables.

**Step 3: Dimensionality Reduction**  
Applied **Principal Component Analysis (PCA)** to reduce dimensionality and eliminate noise, enhancing clustering effectiveness.

**Step 4: Determine Optimal Number of Clusters**  
Used the **Elbow Method** on the PCA-transformed dataset to identify a suitable number of clusters for KMeans.

**Step 5: Apply KMeans Clustering**  
Performed **KMeans clustering** to segment churned users based on behavioral similarities.

**Step 6: Evaluate Clustering Performance**  
Validated the clustering result using the **Silhouette Score** to ensure that the clusters are well-defined and meaningful.  

#### 📊 Cluster Summary  

| Cluster | Key Behaviors |
|--------|----------------|
| **0** | Long-tenured users (Tenure = 10), high order count (14), high cashback (~292), no complaints, long delivery (18 days), churned 9 days after last order. |
| **1** | Very new users (Tenure = 1), low order count (2), low cashback, has complaints, delivery ~13 days, churned the day after purchase. |
| **2** | Similar to Cluster 1 but no complaints; also churned immediately after purchase. |
| **3** | New users with moderate order count (6), has complaints, average delivery time, churned after 7 days. |
| **4** | Mid-tenure users (Tenure = 9), moderate orders (5), decent cashback, no complaints, churned after 7 days. |
| **5** | Very new users (Tenure = 1), few orders (2), has complaints, extremely slow delivery (30 days), churned after 2 days. |  

#### 📌 Key Insights

 **Clusters 1, 2, 5**: Immediate churners — indicate onboarding or first experience failure.
- **Cluster 5**: Delivery delay (30 days) is a red flag — urgent logistics issue.
- **Cluster 0**: High-value loyal customers still churned — possibly due to lack of personalization or better offers elsewhere.
- **Cluster 4**: Quiet leavers — no complaints, good tenure, still churned → lack of re-engagement.
- **Cluster 3**: Multiple orders but with complaints → quality/service issues.

#### ✅ Recommendations  

🔹 Cluster 0 – High-Value Loyal Users
- Offer **personalized reactivation campaigns** and exclusive discounts.
- Create **VIP/Loyalty programs**.
- Send **exit surveys** to collect insights and feedback.

🔹 Clusters 1, 2, 5 – Immediate Churners
- Improve **first-time user onboarding** with welcome offers and guidance.
- Address **customer complaints promptly** (especially for Cluster 5).
- Fix **delivery delays** — especially critical for Cluster 5 (30 days is unacceptable).

🔹 Cluster 3 – Engaged but Dissatisfied Users
- Investigate **service/seller/product quality** issues.
- Provide **“We’re sorry” discounts** or vouchers to regain trust.
- Use feedback-based incentives.

🔹 Cluster 4 – Missed Opportunities
- Trigger **win-back campaigns** after 5–7 days of inactivity.
- Use **personalized product recommendations**.
- Launch **limited-time deals** to create urgency.









