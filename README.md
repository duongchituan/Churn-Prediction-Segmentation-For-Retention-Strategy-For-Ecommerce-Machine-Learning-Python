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
4. [🧮 Train & Apply Churn Prediction Model](#-apply-rfm-model)  

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

## 🧮 Apply RFM Model  

### 📌 What is RFM?

Helps identify the most valuable customers for marketers by analyzing customer behavior.

- **Recency:** When was the customer’s last purchase?

→ Indicates the level of engagement and potential interest. Customers who purchased recently are more likely to respond positively to marketing campaigns and promotions.

- **Frequency:** How often does the customer make a purchase?

→ Measures loyalty and long-term engagement. Frequent buyers tend to have higher retention and can be targeted with loyalty programs or exclusive offers.

- **Monetary:** How much does the customer spend?

→ Reflects the value and revenue contribution of the customer. High spenders contribute significantly to total revenue and may deserve special perks or recognition.

The objective is to segment customers based on their activity and value to facilitate targeted marketing and retention strategies.  

### 🧮 Main Process  

#### Step 1: Calculating the three RFM behavioral metrics  

- **Recency:** Number of days since the customer's most recent purchase up to the analysis date.

- **Frequency:** Total number of orders the customer has placed up to the analysis date.

- **Monetary:** Total revenue the customer has spent up to the analysis date.

![image](https://github.com/user-attachments/assets/a2f64a44-d3ff-4df2-b79f-faf1d29da37b)  

#### Step 2: Scoring customers based on RFM values  

After calculating the Recency, Frequency, and Monetary values, each customer is assigned an R, F, and M score using quintiles — dividing all customers into five equally sized groups for each metric.

**Why use quintiles instead of quartiles or deciles?**  

✅ Balanced granularity: Quintiles provide enough segmentation without overcomplicating the model.

✅ Easy to interpret: A consistent score range from 1 to 5 makes the results intuitive and easy to communicate.

✅ Standardized scoring: All three metrics are mapped to the same scale, making them easy to combine into a unified RFM score.

**How scores are assigned:**  

**Recency (R):** Customers who purchased more recently are considered more engaged.  

→ Customers with the **lowest recency values** (i.e., most recent purchases) receive **R = 5**, and those with the **highest values** receive **R = 1**.

**Frequency (F):** Customers who purchase more often are more loyal.  

→ The **most frequent buyers** receive **F = 5**, and the **least frequent** receive **F = 1**.

**Monetary (M):** Customers who spend more contribute more value.  

→ The **highest spenders** receive **M = 5**, and the **lowest spenders** receive **M = 1**.

![image](https://github.com/user-attachments/assets/258cd7ea-867b-4e9a-ae1a-6e50b99f26a9)  

#### Step 3: Creating the combined RFM score  

After assigning individual scores for **Recency**, **Frequency**, and **Monetary**, the three scores are concatenated into a three-digit string. Each customer is assigned a unique RFM score (e.g., 543, 215, 355) that represents their purchasing behavior.  

![image](https://github.com/user-attachments/assets/9fdc5f59-ee58-40b5-a14a-70d323d1a6e3)  

#### Step 4: Assigning customer segments  

After generating the RFM score (e.g., 543, 215...), each score is matched to a specific customer group based on a predefined segmentation table.

This makes it easy to identify which group each customer belongs to, allowing the business to apply appropriate strategies such as offering promotions, providing personalized care, or encouraging repeat purchases.  

![image](https://github.com/user-attachments/assets/70911ad7-4ad0-42cb-8ad7-22856f49eb0f)

---

## 📊 Visualization & Analysis  

### 1. Customer Segment Distribution

![image](https://github.com/user-attachments/assets/fd272775-091f-4871-b153-6237e07f8ea5)  

### 2. Recency by Segment  

![image](https://github.com/user-attachments/assets/bb5baae4-21be-4e7c-a5b8-44e5552ea045)  

### 3. Frequency by Segment

![image](https://github.com/user-attachments/assets/d2d8ef1f-834f-4013-96b2-486e3f25a6ce)

### 4. Monetary by Segment  

![image](https://github.com/user-attachments/assets/73d36f66-549e-417f-820f-7701f4f692aa)  

This table provides a detailed overview of each customer segment based on their average **Recency**, **Frequency**, and **Monetary** values, along with behavioral interpretation and recommended actions.

| Segment               | Recency (days) | Frequency (orders) | Monetary (£) | Behavior Interpretation                                                                 | Suggested Action                                                   |
|-----------------------|----------------|---------------------|--------------|------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| **Champions**         | 32.6           | 246.4               | 2,640,167    | Recently purchased, very frequent, very high spender → **top-tier customers**           | Offer VIP perks, exclusive deals, loyalty upgrades                 |
| **Loyal**             | 60.2           | 108.8               | 622,787      | Consistently engaged and spending → **loyal and reliable**                              | Maintain relationship, cross-sell, reward with incentives          |
| **Potential Loyalist**| 51.9           | 45.9                | 215,402      | Recently engaged, moderate activity → **potential long-term customer**                  | Nurture with emails, offer product bundles                         |
| **New Customers**     | 49.8           | 9.3                 | 49,708       | Recently made first purchase → **still exploring**                                      | Welcome emails, onboarding flow, personalized recommendations      |
| **Promising**         | 29.2           | 19.0                | 32,642       | Recently returned but still low activity                                                 | Small discounts, showcase trending items                          |
| **Need Attention**    | 52.2           | 52.3                | 215,526      | Previously active, now declining → **signs of disengagement**                           | Send reminder emails, recommend new or trending products           |
| **At Risk**           | 175.4          | 74.6                | 413,276      | Used to buy frequently and spend a lot → **risk of churn**                              | Win-back campaigns, feedback surveys                               |
| **Cannot Lose Them**  | 263.6          | 91.9                | 59,857       | VIP customers who haven’t purchased in a long time → **high retention priority**        | Send personal messages, exclusive win-back offers                  |
| **About To Sleep**    | 101.7          | 20.3                | 24,333       | Low engagement, infrequent purchases → **may become inactive soon**                     | Light re-engagement campaigns, send helpful reminders              |
| **Hibernating**       | 174.0          | 19.8                | 259,601      | Long inactive, low value → **low potential recovery**                                   | Run discount campaigns, consider excluding from future campaigns   |
| **Lost Customers**    | 293.5          | 8.2                 | 50,352       | Almost no recent activity → **very low value**                                          | Exclude from marketing efforts, clean from list                    |                          |

---
## 💡 Insight & Recommendation  

To simplify customer segmentation and enhance strategic planning efficiency, the original RFM segments have been consolidated into broader groups based on behavioral characteristics. Maintaining too many detailed segments can hinder decision-making, disperse resources, and reduce implementation effectiveness. Grouping customers with similar behaviors, engagement levels, and value contributions allows for clearer prioritization and more effective allocation of budgets, personnel, and marketing efforts. This approach not only streamlines the analytical framework but also ensures better alignment between customer value and investment strategy.

#### **1. Core Value Customers**  
**Includes:** Champions, Loyal, Potential Loyalist  
**Reason for grouping:** These customers purchase frequently, spend significantly, and are highly engaged or show strong long-term potential. They generate most of the revenue and should be prioritized for retention and value maximization.  

#### **2. Nurture & Growth**  
**Includes:** New Customers, Promising, Need Attention  
**Reason for grouping:** These customers are either new, recently re-engaged, or starting to decline. They hold potential for growth if nurtured properly through engagement and personalized offers.  

#### **3. At Risk & Retention**  
**Includes:** At Risk, Cannot Lose Them  
**Reason for grouping:** Previously high-value customers who haven't purchased for a while. They are at high risk of churn and require immediate win-back strategies to restore engagement.  

#### **4. Churned / Low Value**  
**Includes:** About To Sleep, Hibernating, Lost Customers  
**Reason for grouping:** These customers have low engagement, low spending, and long inactivity. They offer limited recovery potential and should be deprioritized or removed from active campaigns.  

### 🙋‍♂️ Customer Segmentation Strategy  

#### 1️⃣ Core Value Customers  
**Includes:** Champions, Loyal, Potential Loyalist  
**Traits:** Represent ~40% of customers and contribute the largest share of revenue.  
- Very high purchase frequency, recent transactions (within 30–60 days), and high spending.  
- Act as the financial backbone of the business; highly responsive to upsell and cross-sell.  
- Potential Loyalists show strong signs of future long-term value.

**Recommended Actions:**  
- Provide VIP benefits and loyalty privileges.  
- Encourage reviews or referrals.  
- Prioritize for customer service and new product campaigns.

#### 2️⃣ Nurture & Growth Segment  
**Includes:** New Customers, Promising, Need Attention  
**Traits:** Around 16% of customers; in early-stage engagement or beginning to decline.  
- Recent purchases but irregular, low-to-mid frequency, and modest spending.  
- Can be cultivated into loyal customers with proper engagement.

**Recommended Actions:**  
- Send welcome/onboarding emails.  
- Suggest product bundles or apply light discounts to trigger repeat purchases.  
- Set up inactivity triggers and gentle nudges.

#### 3️⃣ At Risk Segment  
**Includes:** At Risk, Cannot Lose Them  
**Traits:** Nearly 11% of customers; high past value but long periods of inactivity.  
- Previously frequent buyers with high spend, but last purchase was 6+ months ago.  
- May be considering competitors or have unmet needs.

**Recommended Actions:**  
- Launch “We miss you” campaigns with time-limited offers.  
- Personalize messages based on purchase history.  
- Conduct surveys to identify disengagement reasons.

#### 4️⃣ Churned / Low Value Segment  
**Includes:** About to Sleep, Hibernating, Lost Customers  
**Traits:** Roughly 33% of customers with low activity, low spend, and long inactivity.  
- Frequency < 20 orders, Recency 100–290+ days, low monetary value.  
- Limited recovery potential except a few Hibernating cases.

**Recommended Actions:**  
- Try simple reactivation (e.g., free shipping or small voucher).  
- If no response after 1–2 attempts → remove from active marketing list.  

#### ✅ Strategic Summary  
- **Prioritize investment** in **Core Value Customers** to maintain stable revenue and deepen loyalty.  
- **Selectively nurture** the **Nurture & Growth** segment to convert them into high-value customers through personalized engagement.  
- **Implement early warning mechanisms** and recovery plans for the **At Risk** segment to prevent silent churn.  
- **Optimize marketing costs** by **discontinuing investment** in **Churned / Low Value** customers unless specific recovery signals appear.


### 🏢 Business Recommendation  

For a global retail company like SuperStore, which serves a large and diverse customer base, identifying the most impactful metric within the RFM model is essential for effective marketing and sales strategies. Among the three metrics (Recency, Frequency, and Monetary), Recency should be the top priority.

Recency measures how recently a customer made a purchase. This is a strong indicator of engagement. Customers who bought recently are much more likely to respond to marketing campaigns, take advantage of promotions, and make repeat purchases. In contrast, even if a customer has purchased frequently or spent a lot in the past, a long period of inactivity may signal a loss of interest.

By focusing on Recency first, teams can identify active or “warm” customers and allocate resources to those with a higher chance of conversion.


