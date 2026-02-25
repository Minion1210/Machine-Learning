# Machine-Learning__Sleep Disorder Prediction
I. This project aims to develop an efficient, low-cost, and automated prediction system. By leveraging non-invasive physiological data and lifestyle habits through machine learning algorithms, the system provides users with early health warnings and actionable insights into their sleep health.

II. Feature Engineering (Inputs & Outputs)
Key Input Features: The model incorporates a comprehensive range of multi-dimensional parameters:

Demographics & Lifestyle: Sex, Age, Occupation, Physical Activity Level, and subjective Stress Level.

Health Indicators: BMI Category, Blood Pressure, Heart Rate, and Daily Steps.

Sleep Metrics: Sleep Duration and Sleep Quality Rating.

Predicted Outcome (Output):

Target Variable: Identification of Sleep Disorders (Classified as: None or Insomnia).

III. Methodology & Data Preprocessing
Variable Transformation: Categorical data (e.g., Sex, Occupation, BMI Category, and Blood Pressure) were transformed into numerical formats using encoding techniques to ensure full compatibility with machine learning algorithms.

Evaluation Framework:

Data Splitting: A standard 80/20 train-test split was implemented, reserving 20% of the data for independent validation.

Comparative Analysis: Multiple machine learning models were deployed and benchmarked to identify the most robust diagnostic approach.

IV. Results & Discussion
Key Predictors: Feature importance analysis revealed that the top three factors influencing the prediction are BMI Category, Sleep Duration, and Daily Steps.

Model Performance (ANFIS):

The Adaptive Neuro-Fuzzy Inference System (ANFIS) emerged as the most effective classification method, achieving superior predictive accuracy.

Efficiency Trade-off: While ANFIS provided the highest precision, it exhibited higher computational resource consumption compared to RBFNN (Radial Basis Function Neural Network) or ANN (Artificial Neural Network).

V. Practical Implication
A strategic balance between predictive performance and computational efficiency is essential for practical deployment, especially when integrating the model into wearable devices or mobile health applications.
