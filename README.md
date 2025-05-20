### Sales Forecasting in Pharmaceutical Industry

![image](https://github.com/user-attachments/assets/51bdd21f-942a-4aeb-afa7-3cfae94f412a)

This project focuses on developing a time-series based sales forecasting model for pharmaceutical products using historical sales data. The goal is to aid better sales planning, inventory management, and strategy formulation for different drug categories.

**Project Overview**

**Objective:** Analyze historical pharmaceutical sales data and forecast future sales using automated time-series models.

**Scope:** Help stakeholders understand sales patterns, reduce inventory costs, avoid stockouts, and ensure raw material availability.

**Approach:** Data-driven forecasting with a full ML pipeline—from data collection to deployment using Flask.

**Goals**

* Analyze and forecast the sales of 8 different ATC-classified drug categories.

* Recommend marketing and inventory strategies based on insights from trends and seasonality.

* Deploy a predictive model for live use to forecast the next 7 days of sales.

**CRISP-ML(Q) Framework**

Followed the CRISP-ML(Q) process:

1. Business Understanding

2. Data Understanding

3. Data Preparation

4. Modeling

5. Evaluation

6. Deployment

7. Quality Assurance

**Tech Stack**

**Languages:** Python

**Libraries:** AutoTS, pandas, NumPy, Sweetviz, Pandas Profiling, D-Tale

**Data:** MySQL, CSV (5+ years of historical sales data)

**Visualization:** matplotlib, seaborn

**Deployment:** Flask Web Application

**IDE:** Jupyter Notebook, Spyder

**Environment:** Anaconda

**Dataset Overview**

Source: Kaggle + MySQL

Period: Jan 2014 – Oct 2019

Records: 2106 rows

Attributes: 13 (including 8 drug categories such as M01AB, M01AE,N02BA, N02BE, N05B, N05C, R03, R06, date/time fields)

Target Variable: Daily Sales per drug (continuous)

**Exploratory Data Analysis**

* Performed univariate, bivariate, and multivariate analysis

* Automated EDA with tools like Sweetviz, Pandas-Profiling, D-Tale

* Identified trends, seasonality, outliers, and zero-value patterns in data

**Model Building**

* Tried models like ARIMA, SARIMA, and FBProphet

* Final model selection via AutoTS which auto-selects best-fit models from 20+ candidates using genetic programming

* Evaluated using SMAPE (Symmetric Mean Absolute Percentage Error)

**Results**

* Highest Demand Drug: N02BE

* Moderate Demand: M01AB, M01AE, N05B, R03

* Lowest Demand: N05C, R06

Best Forecast Accuracy: Varies per drug, determined by SMAPE

**Deployment**

* Model deployed using a Flask API

* Predicts sales for the next 7 days dynamically

* Future potential for integration with inventory or ERP systems

**Challenges**

* AutoTS has limitations with multivariate time series

* Deployment involved dependency/version mismatches (e.g., statsmodels)

* Difficulty in contextual modeling due to lack of external features

**Future Scope**

* Explore other AutoML time-series libraries (e.g., PyCaret, H2O)

* Integrate external data (e.g., seasonality drivers, marketing data)

* Deploy with CI/CD tools for production use

* Create dashboards using Power BI or Streamlit for better visualization
