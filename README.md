<h1 align='center'>
<u>KAGGLE CREDIT CARD FRAUD DETECTION</u>
</h1>

<img src='./Images/credit_card_protection.jpeg' style='width: 100%; height: 400px; object-fit: cover;' />

## Declaration

> **Author**: This project was independently conceived, developed, and completed by Ngundo Muithya. All code, analysis, and documentation herein represent my own original work.

## Overview

Credit card fraud is a significant and growing threat to the financial industry, both globally and in Kenya. According to the [Central Bank of Kenya's Financial Sector Stability Report](https://businessfront.com/finance/insights/fraud-costs-hit-11m-kenyan-banks/), card fraud losses at Kenyan banks surged more than 16-fold in a single year, from **KSh 15.5 million** to **KSh 263.3 million**, even as the number of reported incidents rose only marginally. This shows that individual fraud cases are becoming far more costly, not just more frequent, underscoring the need for effective detection systems.

Fraudulent transactions also represent a small fraction of all transactions, resulting in highly imbalanced datasets that make detection a challenging machine learning problem. This project builds a classification pipeline to detect fraudulent credit card transactions, addressing this imbalance through techniques such as SMOTE and tuned ensemble models.

## Instructions to clone

Ensure, after cloning the repository, to run the following command in the terminal while in the project folder:

```bash
git lfs pull
```

This downloads the credit card data into your local machine.

<h2 align='center'>
1. Data Understanding
</h2>

The dataset used ([here](./creditcard.csv)) comes from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and contains transactions made by European cardholders in Spetember 2013.

The dataset is highly imbalanced with the positive class (frauds) accounting for only **0.172%** of the transactions.

Features **V1, V2,..., V28** are the principal components obtained with PCA. The only features which have not been transformed with PCA are **Time** and **Amount**.

**Time** contains the **seconds** elapsed between each transaction and the first transaction in the dataset.

**Amount** is the transaction amount, this feature can be used for example-dependant cost-sensitive learning. (The currency was not provided by Kaggle)

**Class** is the response variable and it takes value 1 in case of fraud and 0 otherwise.

<h2 align='center'>
2. Data Visualisation
</h2>

### a) Class Imbalance
