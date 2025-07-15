# Ecommerce-recommendation-system-using-machine-learning-
This project demonstrates a complete machine learning pipeline to predict whether a user will make a purchase based on personalized user features. It includes data cleaning, feature engineering, handling imbalanced data, training a classifier, and evaluating model performance.
Key Techniques and Steps
1. Data Cleaning and Preprocessing
Numeric Conversion & Missing Value Handling:
Numeric columns (e.g., Age, Income, Purchase Frequency) are converted to numeric types using pd.to_numeric(), with invalid values converted to NaN and then replaced by 0 to avoid errors in modeling.

Categorical Encoding:
Textual categorical columns (e.g., Gender, Location, Interests) are transformed into numerical labels using LabelEncoder to prepare for machine learning algorithms which require numeric input.

2. Feature Engineering
Synthetic Feature Creation:
New features such as 'visitor' (indicating whether the user is a new or returning visitor), 'bought_count' (randomized count of items purchased), and 'purchased' (binary target indicating purchase or not) are added to enrich the dataset and provide more signals to the model.

Encoding New Categorical Features:
The new categorical 'visitor' feature is also label encoded for compatibility with ML models.

3. Exploratory Data Analysis
Correlation Analysis with Clustered Heatmap:
To understand relationships among features, a clustered heatmap is generated to visualize the correlation matrix. Hierarchical clustering groups features with similar correlation patterns, helping identify strong relationships and potential redundancies.

4. Feature Scaling
Standardization:
Features are scaled using StandardScaler to normalize the data so each feature has zero mean and unit variance. This is critical for many machine learning algorithms to perform optimally.

5. Handling Imbalanced Data
SMOTE (Synthetic Minority Over-sampling Technique):
SMOTE is applied to balance the target classes by generating synthetic samples for the minority class. This prevents bias toward majority classes and improves generalization.

6. Model Training and Evaluation
Train-Test Split:
The balanced dataset is split into training and testing sets to evaluate the model’s performance on unseen data.

XGBoost Classifier:
An XGBoost model is trained on the training data, known for its high accuracy and efficiency with structured data.

Prediction and Thresholding:
The model outputs probabilities, which are converted to class predictions using a decision threshold of 0.5.

Performance Metrics:
The model is evaluated using accuracy, confusion matrix, precision, recall, and F1-score.

Confusion Matrix Visualization:
A heatmap of the confusion matrix is plotted to visualize true positives, false positives, true negatives, and false negatives, aiding interpretation of model errors.

