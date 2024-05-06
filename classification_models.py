from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from cleaning import * 
from data_visualizations import *
import random
import matplotlib.pyplot as plt 
import seaborn as sns

X = data_without_outliers[['MonthlyCharge', 'DataUsage', 'DayMins', 'DayCalls']].values 
y = data_without_outliers['Churn'].values 

X_train, X_valid, y_train, y_valid = train_test_split(X,y)

# Naive Bayes Classifier 
nbclassifier = GaussianNB()
nbclassifier.fit(X_train, y_train)
print('model score for the training data using Bayes: ', nbclassifier.score(X_train, y_train))
print('model score for testing data using Bayes: ', nbclassifier.score(X_valid, y_valid))

# K-Nearest Neighbours 
knnclassifier = make_pipeline( 
    MinMaxScaler(),
    KNeighborsClassifier(n_neighbors=25)
)
knnclassifier.fit(X_train, y_train)
print('model score for the training data using Knn: ', knnclassifier.score(X_train, y_train))
print('model score for testing data using Knn: ', knnclassifier.score(X_valid, y_valid))

# Decision Tree Classifier 
dtclassifier = DecisionTreeClassifier(max_depth = 9)
dtclassifier.fit(X_train, y_train)
print('model score for the training data using DecisionTrees: ', dtclassifier.score(X_train, y_train))
print('model score for testing data using DecisionTrees: ', dtclassifier.score(X_valid, y_valid))

# Random Forest Classifier 
rfclassifier = RandomForestClassifier(n_estimators=500, max_depth=7) 
rfclassifier.fit(X_train, y_train)
print('model score for the training data using RandomForest: ', rfclassifier.score(X_train, y_train))
print('model score for testing data using RandomForest: ', rfclassifier.score(X_valid, y_valid))

# Gradient Boosted Trees 
gboost = GradientBoostingClassifier(n_estimators=350, max_depth=9, min_samples_leaf=(0.2)) # 0.887, 200, 5, 0.3
gboost.fit(X_train, y_train)
print('model score for the training data using GradientBoosting: ', gboost.score(X_train, y_train))
print('model score for testing data using GradientBoosting: ', gboost.score(X_valid, y_valid))

# confusion matrix for Gradient Boosting classifier 
y_prediction = gboost.predict(X_valid)
conf_matrix = confusion_matrix(y_valid, y_prediction)
sns.set(font_scale= 1.4)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'YlGnBu', xticklabels=gboost.classes_, yticklabels=gboost.classes_)

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Gradient Boosting Classifier')
plt.savefig("Figures/Confusionmatrix_boosting")

### visualization of all validation scores for each model
data = pd.DataFrame({
    'Model Type': ['Bayesian Classifier', 'KNN', 'Decision Trees', 'Random Forest', 'Gradient Boosting'],
    'Validation Score': [nbclassifier.score(X_valid, y_valid),
                            knnclassifier.score(X_valid, y_valid),
                            dtclassifier.score(X_valid, y_valid),
                            rfclassifier.score(X_valid, y_valid),
                            gboost.score(X_valid, y_valid)]
})
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x='Model Type', y='Validation Score', data=data, palette='viridis')
plt.title('Validation Scores for Different Models')
plt.xlabel('Models')
plt.ylabel('Validation Accuracy Scores')

for index, value in enumerate(data['Validation Score']):
    bar_plot.text(index, value + 0.01, f'{value:.2f}', ha='center', va='bottom', fontsize=8)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Figures/Validation_Scores_barplot')