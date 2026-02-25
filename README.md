# Machine-Learning
Machine Learning-Based Approach for Sleep Disorder Prediction

This project aims to develop an efficient and low-cost automated prediction system using non-invasive physiological data and lifestyle habits, leveraging machine learning algorithms to assist users in early health warning.

1. Key Input Features (Inputs)
The model incorporates a comprehensive range of physiological and lifestyle parameters, including:

Demographics: Sex, Age, Occupation, Physical Activity Level, and subjective Stress Level.

Health Indicators: BMI Category, Blood Pressure, Heart Rate, and Daily Steps.

Sleep Metrics: Sleep Duration and Sleep Quality Rating.

2. Predicted Outcome (Output)
Target Variable: Identification of Sleep Disorders (Classified as: None or Insomnia).

3. Data Preprocessing & Methodology
Variable Transformation: Categorical data (such as Sex, Occupation, BMI Category, and Blood Pressure) were transformed into numerical formats using encoding techniques to ensure compatibility with machine learning algorithms.

Evaluation Framework:

Data Splitting: A 80/20 train-test split was implemented, utilizing 80% of the dataset for model training and 20% for independent validation.

Comparative Analysis: Multiple machine learning models were deployed and compared to determine the most effective diagnostic approach.

4. Results and Discussion
Key Predictors: Feature importance analysis revealed that the top three factors influencing the prediction are BMI Category, Sleep Duration, and Daily Steps.

Model Performance (ANFIS):
Efficiency Trade-off: While ANFIS provided the highest precision, it required higher computational resource consumption compared to RBFNN (Radial Basis Function Neural Network) or ANN (Artificial Neural Network). Consequently, a strategic balance between predictive performance and computational efficiency is essential for practical deployment.
