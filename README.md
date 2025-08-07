# 🧠 Time-Series Classification of Human Behavior

## 🏆 Project Overview & Competition Goal
This project was developed for the **Kaggle "CMI Detect Behavior" competition**. The primary objective was to classify complex human gestures, specifically body-focused repetitive behaviors (BFRBs), using multi-modal time-series data captured from a custom wrist-worn sensor. This is a challenging problem with direct applications in digital health and behavior monitoring.

## ✨ Key Achievements & Results
* **Performance:** Achieved a competitive public score of **0.82** on the Kaggle leaderboard, demonstrating the effectiveness of the proposed solution.

## 🛠️ Technical Approach & Tech Stack

This solution employed a multi-faceted approach, combining a novel deep learning architecture with sophisticated feature engineering and a robust training strategy.

#### Model Architecture
A novel **hybrid CNN-Transformer (BERT) architecture** was developed in PyTorch. This hybrid model leverages the strengths of both architectures:
* **CNNs** were used to capture local patterns and features from the raw sensor data sequences.
* **Transformers (BERT)** were used to model long-range dependencies and contextual relationships across the time series.

#### Feature Engineering
A key part of the project's success was the creation of a robust set of custom features. These were not just statistical aggregations but were derived by applying **physics-based principles**, such as **gravity compensation**, to the raw accelerometer and gyroscope data. This provided the model with a richer, more meaningful representation of the user's movements.

#### Final Strategy
The final submission was an **ensemble of 5 models**, each trained using a **5-fold cross-validation strategy**. This technique significantly enhanced the overall prediction accuracy and the solution's robustness.

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
