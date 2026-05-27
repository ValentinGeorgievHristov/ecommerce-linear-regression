# Ecommerce Customer Analytics: Linear Regression Project

## 📌 Project Overview
This data science project focuses on analyzing customer behavior data for an ecommerce clothing company. The business offers both an online store and in-store style consultations. The primary goal is to determine whether the company should focus its development and budget on optimizing its mobile application or its desktop website to maximize yearly revenue.

---

## 🛠️ Technologies Used
* **Python 3**
* **Pandas & NumPy** (Data manipulation and analysis)
* **Matplotlib & Seaborn** (Data visualization and EDA)
* **Scikit-Learn** (Machine Learning model training and evaluation)

---

## 📈 Methodology & Project Phases

### Phase 1: Environment Setup & Library Imports
Configuring the environment and importing data science libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`).

### Phase 2: Data Acquisition & Initial Exploration
Loading the `Ecommerce Customers` dataset and performing baseline statistical checks (`.info()`, `.describe()`) to verify data consistency across all 500 records.

### Phase 3: Exploratory Data Analysis (EDA)
Visualizing data distributions and core relationships using Seaborn charts (`jointplot`, `pairplot`, and `lmplot`). 
* Verified that **Length of Membership** exhibits the strongest visual linear correlation with overall spend, supported by a strong Pearson correlation coefficient of **0.81**.

### Phase 4: Data Splitting (Train-Test Split)
Isolating the independent features matrix ($X$) from the target target variable ($y$). Splitting the rows into **70% Training** and **30% Testing** subsets to ensure unbiased model validation.

### Phase 5: Model Initialization & Training
Instantiating and fitting a `LinearRegression` model using Scikit-Learn to compute the mathematical feature weights.

### Phase 6: Predicting Test Data & Visual Evaluation
Generating predictions on the unseen testing partition ($X_{test}$) and validating model assumptions using:
* A tight, diagonal actual-vs-predicted scatter plot.
* A normally distributed (bell-shaped) residuals histogram centered at zero.

### Phase 7: Performance Metrics & Business Insights
Quantifying error rates using standard regression validation formulas.

---

## 📊 Evaluation Metrics Results
The trained Linear Regression model achieved outstanding accuracy on the test set:
* **Initial MAE (Mean Absolute Error):** \$7.23
* **Initial RMSE (Root Mean Squared Error):** \$8.93
* **5-Fold Cross-Validated RMSE (Final Audit):** \$10.02

### 🔍 Bias-Variance & Robustness Analysis
By comparing our initial test RMSE (\$8.93) with the 5-Fold Cross-Validated RMSE (\$10.02), we observe an exceptionally narrow gap (~\$1.09). This serves as mathematical proof that the model has **low variance** and does not suffer from overfitting. The 5-Fold cross-validation utilized a strict **4:1 nested proportion** (400 rows for training, 100 rows for testing), rotating 5 times so that **every single row was tested exactly once**. This eliminates any "shadow" (uninspected) data, confirming a highly robust **~2% true error rate** (approx. **98% overall accuracy**) under simulated real-world conditions.


---

## 💡 Strategic Business Recommendations
Based on the extracted model coefficients and verified by the cross-validation audit confirming a ~98% overall accuracy, we provided the following data-driven insights:

1. **App vs. Website:** Time spent on the Mobile App yields **\$38.59** per unit of time, whereas time spent on the Website desktop platform yields only **\$0.19**. Website behavior has virtually zero impact on the revenue.
2. **Action Plan:** The company should shift its primary engineering focus and budget away from website desktop updates and heavily invest in **optimizing the mobile application experience**.
3. **The Customer Retention Driver:** **Length of Membership** has the single highest financial impact overall, contributing an extra **\$61.28** annually per year of loyalty. Launching automated retention campaigns and membership rewards is highly recommended to increase lifetime value.


