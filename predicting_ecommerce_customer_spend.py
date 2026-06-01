#!/usr/bin/env python
# coding: utf-8

# # Ecommerce Customers Analysis: Linear Regression Project
# 
# ## Project Overview
# This project analyzes customer behavior data for an e-commerce company that sells clothing online but also has in-store style and clothing advice sessions. The goal is to help the company decide whether to focus their efforts on their mobile app experience or their website desktop platform to maximize yearly revenue.
# 
# ---
# ### Phase 1: Environment Setup & Library Imports
# We begin by importing the essential libraries for data analysis, numerical computing, and data visualization.
# 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure that all matplotlib plots are displayed inline within the Jupyter Notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Phase 2: Data Acquisition & Initial Exploration
# 
# We read the customer dataset into a Pandas DataFrame and perform baseline statistical checks (`.head()`, `.info()`, and `.describe()`) to understand the data structure and verify cleanliness.
# 

# In[3]:


customers = pd.read_csv('Ecommerce Customers')


# In[4]:


customers.head()


# In[5]:


customers.describe()


# In[6]:


customers.info()


# ### Phase 3: Exploratory Data Analysis (EDA)
# 
# We utilize Seaborn to visually inspect the relationships between customer behaviors (Time on App, Time on Website, Length of Membership) and the target variable (`Yearly Amount Spent`) to identify strong linear correlations.
# 

# **1. Analyzing Time on Website vs. Yearly Amount Spent**
# We check if customers who spend more time browsing the desktop website tend to spend more money.
# 

# In[7]:


sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers)


# **2. Analyzing Time on App vs. Yearly Amount Spent**
# We perform the same analysis for the mobile application to compare its direct impact on sales against the desktop website.

# In[8]:


sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers)


# **3. Density Analysis (Hex Bin Plot)**
# We compare the interaction between Time on Website and Length of Membership using a 2D hex bin plot to visualize where the highest concentration of customers is located.

# In[9]:


sns.jointplot(x='Time on Website', y='Length of Membership', data=customers, kind='hex')


# **4. Entire Dataset Relationships (Pairplot)**  
# We explore the relationships across all numerical features in the dataset simultaneously. This matrix of plots allows us to visually detect which specific metric shares the strongest correlation with our target variable.

# In[10]:


sns.pairplot(customers)


# **5. Statistical Correlation Matrix**  
# To validate our visual findings from the pairplot with exact numbers, we calculate the Pearson correlation coefficients for all numerical features. This matrix confirms the statistical strength of each relationship.
# 

# In[11]:


customers.corr(numeric_only=True)


# **6. Visualizing the Strongest Linear Correlation (Lmplot)**  
# Based on the correlation matrix and pairplot, `Length of Membership` shows the most dominant relationship with `Yearly Amount Spent`. We create a linear model plot to draw a regression line through the data and visualize this tight fit.
# 

# In[12]:


sns.lmplot(x='Length of Membership', y='Yearly Amount Spent',data=customers)


# **Feature Identification (Column Inspection)**  
# 
# Before dividing the data into independent features (X) and the target variable (y), we inspect the exact names of all columns in our dataset. This ensures we select only the relevant numerical metrics for our machine learning model.
# 

# In[13]:


# Display all available columns in the dataframe
customers.columns


# ### Phase 4: Data Splitting (Train-Test Split)
# 
# To evaluate our model properly and ensure it can generalize well to new customers, we must split our data. We first isolate our predictive features from our target goal, and then divide them into training and testing subsets.
# 
# **1. Defining Features (X) and Target Variable (y)**
# We extract only the numerical columns that capture customer behaviors into our features matrix (`X`), and isolate the `Yearly Amount Spent` as our target variable (`y`). We intentionally exclude text-based columns like Email and Address, as they do not provide mathematical value to a linear regression model.
# 

# In[14]:


X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

y = customers['Yearly Amount Spent']


# **2. Executing the Train-Test Split**
# 
# We use `model_selection.train_test_split` from Scikit-Learn to split the data into training and testing sets. By setting `test_size=0.3`, we allocate 30% of our data for evaluation, while `random_state=101` ensures the shuffle is reproducible.
# 

# In[15]:


from sklearn.model_selection import train_test_split

# Split the features and target matrices into 70/30 subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ### Phase 5: Model Initialization & Training
# 
# We import the `LinearRegression` algorithm from Scikit-Learn and initialize a model instance. 
# Then, we fit the model using the training data to calculate the optimal weights (coefficients).

# In[17]:


from sklearn.linear_model import LinearRegression

# Instantiate the model (create the empty machine)
lm = LinearRegression()

# Train the model using the training datasets (solve the equation)
lm.fit(X_train, y_train)


# In[18]:


# Print the raw array of coefficients
print('Coefficients: \n', lm.coef_)


# **Model Coefficients & Feature Interpretation**
# 
# After training the model, we extract the coefficients to evaluate how much each feature impacts the `Yearly Amount Spent`.  
# To make the data highly readable, we align the features with their calculated weights in a Pandas DataFrame.
# 
# > *Business Insight:* Look closely at the comparison between the Website and the App.  
# The App generates **$38.59** per unit of time, while the Website brings in practically nothing (**$0.19**).  
#                                                                                                  This clearly demonstrates that mobile optimization should be the company's top strategic priority.
# 

# In[33]:


import pandas as pd

# Create a clean dataframe linking feature names to their coefficients
coefficients_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficients'])
coefficients_df


# ### Phase 6: Predicting Test Data & Visual Evaluation
# 
# Now that the model is trained, we evaluate its predictive power on unseen data. We pass the testing features (`X_test`) through the model to generate forecasts and visually check the quality of the predictions.
# 
# **1. Generating Predictions**
# We use the `.predict()` method to calculate the estimated yearly spend for the 150 customers in our testing set.
# 
# 

# In[ ]:


# Generate predictions for the unseen test features
predictions = lm.predict(X_test)


# **Visualizing Predictions (Real vs. Predicted Values)**
# 
# We create a scatter plot to visually compare the actual target values (`y_test`)  
# against the model's predictions. The closer the data points are to a tight, straight  
# diagonal line, the higher the accuracy of our linear model.
# 

# In[36]:


import matplotlib.pyplot as plt

# Create the scatter plot to compare real and predicted values
plt.scatter(y_test, predictions, edgecolors='white', alpha=0.7)

# Add clear business labels and titles
plt.xlabel('Y Test (True Values)')
plt.ylabel('Predicted Values')
plt.title('Model Predictions Evaluation')

# Optimize the padding around the plot labels
plt.tight_layout()


# **Evaluating Residuals (Errors Distribution)**
# 
# We plot a histogram of the residuals (the differences between actual and predicted values) to check if they are normally distributed. A symmetric, bell-shaped curve centered around zero indicates that our model's errors are purely random, confirming that the linear regression assumptions hold true.
# 

# In[37]:


import seaborn as sns

# Plot the distribution of the prediction errors
sns.histplot(y_test - predictions, kde=True)


# **Model Evaluation Metrics**
# 
# To quantify the model's accuracy, we calculate the three standard regression metrics:
# * **MAE (Mean Absolute Error):** The average absolute distance between the predictions and true values.
# * **MSE (Mean Squared Error):** Punishes larger, outlier errors by squaring them.
# * **RMSE (Root Mean Squared Error):** The gold standard metric, converting the squared errors back into the original unit (dollars).

# In[38]:


from sklearn import metrics
import numpy as np

# Print the final regression evaluation metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# **3. Model Robustness via K-Fold Cross Validation**
# 
# To ensure that our initial test results (RMSE = $8.93) were not driven by a fortunate random split of the data (High Variance), we implement a 5-Fold Cross Validation. This process splits our dataset into 5 equal parts, rotates the test set 5 times, and returns an honest, averaged performance score.
# 

# In[21]:


from sklearn.model_selection import cross_val_score

# 1. Calculate the negative mean squared error across 5 individual rounds
cv_mse_scores = cross_val_score(lm, X, y, cv=5, scoring='neg_mean_squared_error')

# 2. Convert the negative score outputs into positive values
cv_mse_scores = -cv_mse_scores

# 3. Take the square root of each fold to convert the error back into clean dollars (RMSE)
cv_rmse_scores = np.sqrt(cv_mse_scores)

# 4. Print the individual fold outcomes and the final arithmetic average
print("RMSE scores for the 5 individual folds:\n", cv_rmse_scores)
print("\nFinal Averaged Cross-Validated RMSE:", cv_rmse_scores.mean())


# **4. Architectural Breakdown of the 5-Fold Cross Validation**
# 
# To truly understand what happens during this validation process, we can break down its internal mechanics:
# * **The 4:1 Nested Proportion:** By setting `cv=5`, the algorithm automatically segments the entire 500-row dataset into 5 equal subsets of 100 rows each. It then executes 5 consecutive nested Linear Regressions. In each individual round, the model uses a strict **4:1 proportion**—utilizing 4 folds (400 rows / 80%) for training and 1 fold (100 rows / 20%) for testing.
# * **Eliminating "Shadow" Data:** In a standard single `train_test_split`, the training data remains in the "shadows"—it is never officially tested, meaning hidden overfitting could go unnoticed. Through 5 rounds of rotation, **every single row of our data goes to the exam exactly once**. This guarantees that no data remains uninspected, exposing the true, unvarnished stability of our algorithm.
# 
# The fact that our Cross-Validated RMSE (**$10.02**) is exceptionally close to our initial test RMSE (**$8.93**) is mathematical proof that the model possesses a low variance, is highly robust, and is ready for real-world deployment.
# 

# ### Final Conclusion & Business Recommendation
# 
# Our initial Linear Regression model achieved an outstanding test **RMSE of $8.93**. To strictly validate its robustness, we performed a 5-Fold Cross Validation, which yielded a highly stable and reliable **final averaged RMSE of $10.02**. This confirms that the model generalizes exceptionally well to unseen data and does not suffer from high variance or overfitting. 
# 
# Given that the average yearly customer spend is **$499.31**, the model estimates customer behavior with an exceptional **~2% error rate** (approx. **98% overall accuracy**).
# 
# **Strategic Business Decisions for the Company:**
# 1. **App vs. Website Platform:** Time spent on the Mobile App yields a high coefficient of **$38.59** per unit of time, whereas time spent on the desktop Website yields only **$0.19**. Website desktop traffic has virtually zero material impact on customer spending.
# 2. **Resource Allocation:** The company should immediately stop heavy investments in the desktop website and focus its engineering budget on **optimizing and scaling the mobile application experience**.
# 3. **The Customer Retention Driver:** **Length of Membership** exhibits the single highest financial impact overall, contributing an extra **$61.28** annually per year of customer loyalty. The business must launch aggressive loyalty programs and retention campaigns to maximize customer lifetime value.
# 
# 
