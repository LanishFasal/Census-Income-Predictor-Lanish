# 🧠 Income Prediction Using Census Data

Welcome to the **Income Prediction ML Project Assignment**!  
This is a web application that predicts whether a person earns more or less than $50K annually based on features like age, education, capital gain/loss, and marital status. The application uses a trained machine learning model and offers both a Flask-based web interface and a Streamlit interface for interaction.

## 🎯 Objective

- Deploy the model using both:
  - A **Flask web application**
  - A **Streamlit app**
---

## 📦 Dataset Information

- **Dataset Name:** Adult Census Income
- **Source:** [OpenML](https://www.openml.org/d/1590)
- **Load using:**  
  ```python
  from sklearn.datasets import fetch_openml
  data = fetch_openml("adult", version=2, as_frame=True)
  ````

* **Target column:** `income`

