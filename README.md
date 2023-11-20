# Credit-Card-Fraud-Detection
## Introduction 
This credit card fraud detection project employs machine learning algorithms such as Random Forest, SVM, Decision Trees, k-NN, and Logistic Regression to automatically identify and safeguard against unauthorized transactions. By analyzing transaction data, the goal is to create a precise and efficient system, enhancing the security and reliability of electronic payments, minimizing financial losses due to fraud.

## Features 
- Machine Learning Algorithms
- Testing and Evaluation
- Data Analysis
- Feature Generation

## Architecture Diagram 
![Archi](https://github.com/Meenakshi0907/Credit-Card-Fraud-Detection/assets/94165108/612d2ef5-ad35-467d-95e5-aefeea5a1f80)

## Code Samples 
```py
## Address class imbalance in a dataset
# Define the SMOTE and RandomUnderSampler with specified sampling strategies
over = SMOTE(sampling_strategy=0.5)
under = RandomUnderSampler(sampling_strategy=0.1)

# Extract features and target variable from the first dataset (df1)
f1 = df1.iloc[:, :9].values
t1 = df1.iloc[:, 9].values

# Define the steps for the imbalanced pipeline
steps = [('under', under), ('over', over)]

# Create an imbalanced pipeline with under-sampling followed by over-sampling
pipeline = Pipeline(steps=steps)

# Apply the pipeline to the first dataset (df1)
f1, t1 = pipeline.fit_resample(f1, t1)

# Print the count of each class in the target variable after applying the pipeline
print(Counter(t1))

# Repeat the same process for the second dataset (df2)
over = SMOTE(sampling_strategy=0.5)
under = RandomUnderSampler(sampling_strategy=0.1)

f2 = df2.iloc[:, :20].values
t2 = df2.iloc[:, 20].values

steps = [('under', under), ('over', over)]
pipeline = Pipeline(steps=steps)

f2, t2 = pipeline.fit_resample(f2, t2)

print(Counter(t2))

## Splitting X and Y for Training & Testing

x_train1, x_test1, y_train1, y_test1 = train_test_split(f1, t1, test_size = 0.20, random_state = 2)
x_train2, x_test2, y_train2, y_test2 = train_test_split(f2, t2, test_size = 0.20, random_state = 2)

## Define Functions for Model & Evaluation
def model(classifier, x_train, y_train, x_test, y_test):
    # Fit the classifier on the training data
    classifier.fit(x_train, y_train)

    # Make predictions on the test data
    prediction = classifier.predict(x_test)

    # Perform cross-validation and print the mean ROC AUC score
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    print("Cross Validation Score : ", '{0:.2%}'.format(cross_val_score(classifier, x_train, y_train, cv=cv, scoring='roc_auc').mean()))

    # Print the ROC AUC score on the test data
    print("ROC_AUC Score : ", '{0:.2%}'.format(roc_auc_score(y_test, prediction)))

    # Calculate and print accuracy on the test data
    accuracy = accuracy_score(y_test, prediction)
    print("Accuracy: ", '{0:.2%}'.format(accuracy))

def model_evaluation(classifier, x_test, y_test):
    # Confusion Matrix
    cm = confusion_matrix(y_test, classifier.predict(x_test))
    names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    counts = [value for value in cm.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names, counts, percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cm, annot=labels, cmap='Blues', fmt='')

    # Classification Report
    print(classification_report(y_test, classifier.predict(x_test)))
````
### Logistic Regression
```py
classifier_lr = LogisticRegression(random_state = 0,C=10,penalty= 'l2')

model(classifier_lr,x_train1,y_train1,x_test1,y_test1)
model_evaluation(classifier_lr,x_test1,y_test1)

model(classifier_lr,x_train2,y_train2,x_test2,y_test2)
model_evaluation(classifier_lr,x_test2,y_test2)
```
### Support Vector Machines (SVC)
```py
classifier_svc = SVC(kernel = 'linear',C = 0.1)

model(classifier_svc,x_train1,y_train1,x_test1,y_test1)
model_evaluation(classifier_svc,x_test1,y_test1)

model(classifier_svc,x_train2,y_train2,x_test2,y_test2)
model_evaluation(classifier_svc,x_test2,y_test2)
```
### Decision Tree
```py
classifier_dt = DecisionTreeClassifier(random_state = 1000,max_depth = 4,min_samples_leaf = 1)

model(classifier_dt,x_train1,y_train1,x_test1,y_test1)
model_evaluation(classifier_dt,x_test1,y_test1)

model(classifier_dt,x_train2,y_train2,x_test2,y_test2)
model_evaluation(classifier_dt,x_test2,y_test2)
```
### Random Forest
```py
classifier_rf = RandomForestClassifier(max_depth = 4,random_state = 0)

model(classifier_rf,x_train1,y_train1,x_test1,y_test1)
model_evaluation(classifier_rf,x_test1,y_test1)

model(classifier_rf,x_train2,y_train2,x_test2,y_test2)
model_evaluation(classifier_rf,x_test2,y_test2)
```
### K Nearest Neighbors
```py
classifier_knn = KNeighborsClassifier(leaf_size = 1, n_neighbors = 3,p = 1)

model(classifier_knn,x_train1,y_train1,x_test1,y_test1)
model_evaluation(classifier_knn,x_test1,y_test1)

model(classifier_knn,x_train2,y_train2,x_test2,y_test2)
model_evaluation(classifier_knn,x_test2,y_test2)

```
## Output 
### Correlation Chart for the Dataset
![image](https://github.com/Meenakshi0907/Credit-Card-Fraud-Detection/assets/94165108/a573ff62-0690-4221-8dff-129afe7e73f4)

### Percentage of fraud and non-fraud cases
![image](https://github.com/Meenakshi0907/Credit-Card-Fraud-Detection/assets/94165108/d8bfa67b-583c-4892-a478-c32b03e225a6)

### Logistic Regression 
![image](https://github.com/Meenakshi0907/Credit-Card-Fraud-Detection/assets/94165108/4c325054-40ba-4157-9441-271fe3b1cc99)

### Support Vector Machines (SVC)
![image](https://github.com/Meenakshi0907/Credit-Card-Fraud-Detection/assets/94165108/6112a9c9-7655-41bf-b340-62639620f91c)

### Decision Tree
![image](https://github.com/Meenakshi0907/Credit-Card-Fraud-Detection/assets/94165108/6aff9040-5ca1-4094-9d00-1d9c1dac7921)

### Random Forest
![image](https://github.com/Meenakshi0907/Credit-Card-Fraud-Detection/assets/94165108/833cd941-0905-417b-bddf-1ae953ada23f)

### K Nearest Neighbors
![image](https://github.com/Meenakshi0907/Credit-Card-Fraud-Detection/assets/94165108/22fa2a8e-c937-49b4-a29b-794501db63b1)

### Comparison
![image](https://github.com/Meenakshi0907/Credit-Card-Fraud-Detection/assets/94165108/cac1d23e-b109-4053-8838-8f9fe755d753)

## Result
This credit card fraud detection project, employing algorithms like Random Forest, SVM, Decision Trees, k-NN, and Logistic Regression, aims to automate the identification of normal and fraudulent transactions. Through dataset analysis, training, and testing, the system achieves precision in spotting fraud, enhancing the security of credit card transactions. The project's insights have the potential to mitigate financial losses due to fraud, showcasing the efficacy of smart computer systems in fortifying financial security.
