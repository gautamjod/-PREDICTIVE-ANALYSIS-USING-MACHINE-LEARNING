# -PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING

COMPANY : CODETECH IT SOLUTIONS

NAME : GAUTAM VAID

INTERN ID : CT04DA594

DOMAIN : DATA ANALYTICS

DURATION : 4 WEEKS

MENTOR : NEELA SANTHOSH

🎯 Objective
The primary objective of this project is to develop a supervised machine learning model to accurately classify breast tumors as malignant or benign based on a range of diagnostic measurements derived from digitized images of breast mass tissue. This can serve as a decision-support tool for healthcare professionals to aid early diagnosis and treatment planning.

📚 Dataset Overview
Dataset Used: Breast Cancer Wisconsin (Diagnostic) Dataset (via sklearn.datasets)

Number of Samples: 569

Number of Features: 30 (all numeric and continuous)

Target Variable:

0 – Malignant

1 – Benign

Each feature represents a summary statistic (mean, standard error, and worst) of characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

🔧 Methodology
1. Data Loading and Cleaning
Loaded the dataset into a Pandas DataFrame.

Verified there were no missing values or categorical variables.

Converted the target into a separate Series for supervised learning.

2. Feature Selection
Used SelectKBest with ANOVA F-value (f_classif) to extract the top 10 features most correlated with the target.

This helped reduce overfitting risk and increased interpretability.

Selected Features Included:

mean radius

mean texture

mean perimeter

mean area

worst radius

worst texture

worst perimeter

worst area

worst concave points

worst compactness

3. Data Preprocessing
Split the dataset into training (70%) and testing (30%) using train_test_split.

Standardized features using StandardScaler to normalize scales, which is crucial for many ML algorithms, including those relying on distance metrics.

4. Model Selection and Training
Chose Random Forest Classifier due to:

Strong performance on classification problems

Low risk of overfitting due to ensemble averaging

Ability to rank feature importance

Trained the model on the scaled training dataset using default hyperparameters for initial experimentation.

5. Model Evaluation
Evaluated the model’s performance on the test set using:

Confusion Matrix

Accuracy Score

Classification Report (precision, recall, F1-score)

Results:

Accuracy: ~97%

Precision, Recall, F1-score (Benign): ~0.98

Precision, Recall, F1-score (Malignant): ~0.96

Visualized the confusion matrix using a Seaborn heatmap.

📈 Key Insights
The model performs well in distinguishing between malignant and benign tumors.

Certain features (like worst perimeter and worst concave points) have very strong predictive power.

Feature selection not only improved performance but also helped reduce computational overhead
