# 🛡️ CoverIQ — Insurance Cost Predictor

### Predict Medical Insurance Charges Using Machine Learning

CoverIQ is a machine learning web application that estimates individual medical insurance charges based on personal and lifestyle-related information. It provides an interactive interface for users to enter their details, receive estimated insurance costs, and explore model performance.

---

## 📌 Table of Contents

* [Overview](#-overview)
* [Objectives](#-objectives)
* [Features](#-features)
* [Technology Stack](#-technology-stack)
* [Machine Learning Models](#-machine-learning-models)
* [Model Evaluation](#-model-evaluation)
* [Application Screenshots](#-application-screenshots)
* [Project Structure](#-project-structure)
* [Installation and Setup](#-installation-and-setup)
* [How to Use](#-how-to-use)
* [Dataset](#-dataset)
* [Future Enhancements](#-future-enhancements)
* [Disclaimer](#-disclaimer)

---

## 📖 Overview

Medical insurance charges can vary depending on several factors, including age, body mass index (BMI), smoking habits, number of dependents, and geographical region.

CoverIQ applies machine learning techniques to estimate medical insurance charges using these input features.

The project compares multiple regression algorithms and presents model evaluation metrics to help users understand their predictive performance.

**Project Type:** Machine Learning and Web Application
**Domain:** Healthcare Analytics and Insurance
**Application Framework:** Streamlit
**Problem Type:** Regression

---

## 🎯 Objectives

* Explore and preprocess a medical insurance cost dataset.
* Handle categorical features using appropriate encoding techniques.
* Apply feature scaling where required.
* Train and compare multiple machine learning regression models.
* Evaluate models using MAE, RMSE, and R².
* Perform five-fold cross-validation.
* Develop an interactive web application using Streamlit.
* Allow users to enter personal details and obtain estimated insurance charges.

---

## ✨ Features

### 🏠 Interactive Home Page

A user-friendly landing page introducing CoverIQ and its purpose.

### 🔐 Login Interface

A login interface for accessing the application.

### 💰 Insurance Cost Prediction

Enter relevant personal and lifestyle details to obtain a predicted medical insurance charge.

### 📊 Model Performance Comparison

Compare Linear Regression, Random Forest, and Gradient Boosting using regression evaluation metrics.

### 📈 Model Evaluation

Review MAE, RMSE, R², and five-fold cross-validation results to understand model performance.

### 🧾 Prediction History

Review previously saved predictions, if enabled in the application.

---

## 🛠️ Technology Stack

| Technology          | Purpose                               |
| ------------------- | ------------------------------------- |
| Python              | Main programming language             |
| Pandas              | Data manipulation and preprocessing   |
| NumPy               | Numerical computations                |
| Scikit-learn        | Machine learning and model evaluation |
| Streamlit           | Interactive web application           |
| Matplotlib / Plotly | Visualizations, where implemented     |
| Git                 | Version control                       |
| GitHub              | Source code hosting                   |

*The libraries listed should match the actual imports and implementation in the project.*

---

## 🤖 Machine Learning Models

CoverIQ compares the following regression algorithms.

### 1. Linear Regression

A baseline regression algorithm that models the relationship between input features and insurance charges.

### 2. Random Forest Regressor

An ensemble learning algorithm that combines predictions from multiple decision trees to model complex relationships in the data.

### 3. Gradient Boosting Regressor

An ensemble algorithm that builds decision trees sequentially, with each new tree helping reduce errors made by the existing model.

---

## 📊 Model Evaluation

The models are evaluated using the following metrics.

| Metric                  | Description                                                                                         |
| ----------------------- | --------------------------------------------------------------------------------------------------- |
| MAE                     | Mean Absolute Error; measures the average absolute difference between actual and predicted charges. |
| RMSE                    | Root Mean Squared Error; penalizes larger prediction errors more heavily.                           |
| R² Score                | Measures how well the model explains the variation in the target values.                            |
| 5-Fold Cross-Validation | Evaluates model performance across five data folds to assess consistency.                           |

### Model Comparison Results

Replace the placeholders below with the actual values displayed by your application.

| Model                       |        MAE |       RMSE |   R² Score | Mean 5-Fold CV R² |
| --------------------------- | ---------: | ---------: | ---------: | ----------------: |
| Linear Regression           | Add result | Add result | Add result |        Add result |
| Random Forest Regressor     | Add result | Add result | Add result |        Add result |
| Gradient Boosting Regressor | Add result | Add result | Add result |        Add result |

**Note:** Lower MAE and RMSE generally indicate smaller prediction errors. Higher R² scores generally indicate better fit. Cross-validation results provide additional information about model performance across different data splits.

---

## 📸 Application Screenshots

### 1. Home Page

### 2. Login Page

### 3. Insurance Cost Prediction

### 4. Model Performance Comparison

---

## 📂 Project Structure

```text
CoverIQ-Insurance-Cost-Predictor/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
└── screenshots/
    ├── home_page.png
    ├── login.png
    ├── prediction.png
    └── model_performance.png
```

The `screenshots` folder contains the application images displayed in this README.

---

## ⚙️ Installation and Setup

Follow these steps to run CoverIQ on your local machine.

### Prerequisites

Install the following:

* Python 3.10 or another version compatible with the project's dependencies.
* Git.
* A web browser.

### Step 1: Clone the Repository

```bash
git clone https://github.com/Avani376/CoverIQ-Insurance-Cost-Predictor.git
```

### Step 2: Navigate to the Project Folder

```bash
cd CoverIQ-Insurance-Cost-Predictor
```

### Step 3: Create a Virtual Environment

**Windows:**

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can install the dependencies using the virtual environment's Python executable directly.

### Step 4: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

Streamlit will provide a local URL, typically:

```text
http://localhost:8501
```

Open that address in your browser to access CoverIQ.

---

## 🧭 How to Use

1. Open the CoverIQ application.
2. Explore the home page.
3. Access the login interface if required.
4. Navigate to the insurance prediction section.
5. Enter the requested personal and lifestyle information.
6. Submit the inputs to generate an estimated insurance charge.
7. Visit the Model Performance section to compare the regression models.
8. Review the available prediction history, if enabled.

---

## 📚 Dataset

This project is based on the **Medical Cost Personal Dataset**, commonly used for medical insurance cost prediction.

Typical features include:

| Feature    | Description                                      |
| ---------- | ------------------------------------------------ |
| `age`      | Age of the individual                            |
| `sex`      | Sex category                                     |
| `bmi`      | Body Mass Index                                  |
| `children` | Number of dependents                             |
| `smoker`   | Smoking status                                   |
| `region`   | Residential region                               |
| `charges`  | Medical insurance charges; the prediction target |

The dataset is used to explore relationships between the input features and medical insurance charges.

**Dataset source:** Add the original dataset source or download reference used in your project.

---

## 🔮 Future Enhancements

* Deploy the application to Streamlit Community Cloud.
* Improve model performance through hyperparameter tuning.
* Add additional visualizations for model comparison.
* Enhance prediction explanations.
* Improve application accessibility and mobile responsiveness.
* Add secure user authentication and persistent prediction storage where appropriate.

---

## ⚠️ Disclaimer

CoverIQ provides machine learning-based estimates for educational and demonstration purposes. Predictions are not official insurance quotations and should not be used as the sole basis for financial, medical, or insurance decisions.

Actual insurance charges may vary depending on insurer policies, coverage, location, and other factors.

---

##

---

⭐ If you find this project interesting, consider giving the repository a star!
