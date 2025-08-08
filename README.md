# 🧠 Time-Series Classification of Human Behavior

## 🏆 Project Overview & Competition Goal
This project was developed for the **Kaggle "CMI Detect Behavior" competition**. The primary objective was to classify complex human gestures, specifically body-focused repetitive behaviors (BFRBs), using multi-modal time-series data captured from a custom wrist-worn sensor. This is a challenging problem with direct applications in digital health and behavior monitoring.

## ✨ Key Achievements & Results
* **Performance:** Achieved a competitive public score of **0.82** on the Kaggle leaderboard, demonstrating the effectiveness of the proposed solution.

## 🛠️ Technical Approach & Tech Stack

This solution's success came from a practical and effective strategy: leveraging high-performing public models and enhancing them through custom feature engineering and a robust ensembling technique.

#### Model Strategy
Instead of building a model from scratch, this approach involved a careful selection and fine-tuning of **two high-performing public models** shared within the Kaggle community. My primary contribution focused on:
* **Hyperparameter Tuning:** Systematically tuning the models' parameters to optimize their performance on this specific dataset.
* **Feature Experimentation:** Testing different combinations of features to identify the most predictive inputs for each model.
* **Ensembling:** Developing an effective ensembling strategy to combine the strengths and predictions of both fine-tuned models, which led to a significant boost in the final score.

#### Feature Engineering
A key part of the project's success was the creation of a robust set of custom features. These were not just statistical aggregations but were derived by applying **physics-based principles**, such as **gravity compensation**, to the raw accelerometer and gyroscope data. This provided the models with a richer, more meaningful representation of the user's movements.

#### Tech Stack
* **Languages & Core Libraries:** Python, NumPy, Pandas, Polars
* **Deep Learning:** PyTorch, Transformers (BERT)
* **Machine Learning:** Scikit-learn
* **Platforms:** Kaggle API, Jupyter Notebooks

## 💾 Data
The dataset for this project is from the **Kaggle "CMI Detect Behavior" competition**. Due to its size and competition rules, it is not included in this repository.

**To run this project, you must download the data from the competition page:**
1.  **Download from Kaggle:** You will need a Kaggle account to download the data from the official competition page.
2.  **Place the Data:** Create a folder named `data` in the root of this project and place the downloaded files inside.

## 📂 Project Structure
/human-behavior-classification
|
├── data/                  # Folder for the competition data
├── notebooks/             # Jupyter notebook for EDA and final reporting
├── src/                   # Source code (.py scripts for feature engineering, model definition, training)
├── README.md              # You are here!
└── requirements.txt       # List of Python libraries needed to run the project
