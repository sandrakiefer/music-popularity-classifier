import locale
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import scikitplot as skplt

locale.setlocale(locale.LC_NUMERIC, 'pl_PL')


# ---------------------------------------------------------------------------- #
#                              Data Pre-Processing                             #
# ---------------------------------------------------------------------------- #

def read_data(path):
    """
    Read data from csv file in the given path

    :param path: Path of the CSV file to be imported
    :type path: str
    :return: Dataframe with the imported data
    :rtype: pandas.DataFrame
    """
    df = pd.read_csv(path, index_col=0)
    df.dropna(axis=0, inplace=True)
    df.duplicated().sum()
    return df


# ---------------------------------------------------------------------------- #
#                              Correlation Heatmap                             #
# ---------------------------------------------------------------------------- #

def correlation_heatmap(df):
    """
    Create a correlation heatmap for the given dataframe and save it as a png file

    :param df: Dataframe to be used for the correlation heatmap
    :type df: pandas.DataFrame
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Calculate correlations
    correlations = df[numeric_columns].corr()

    # Create correlation heatmap
    fig = ff.create_annotated_heatmap(
        z=correlations.values,
        x=numeric_columns,
        y=numeric_columns,
        annotation_text=correlations.round(2).values,
        colorscale='Blackbody',
        showscale=True,
        hoverinfo='z'
    )
    fig.update_xaxes(title_text='Features', side='bottom')
    fig.update_yaxes(title_text='Features', side='left')
    fig.update_layout(
        title='Correlation Heatmap',
        title_x=0.5,
        width=1000,
        height=800,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        margin=dict(l=150, r=100, t=100, b=100)
    )
    fig.write_image("PredictPopularity/plots/Correlation_Heatmap.png")


# ---------------------------------------------------------------------------- #
#         Try to predict Views, Likes, Comments, and Streams of a song         #
# ---------------------------------------------------------------------------- #

# ----------------------------- Linear Regression ---------------------------- #

def linear_regression(df, metrics, label):
    """
    Perform linear regression on the given dataframe and save the results in a txt file

    :param df: Dataframe to be used for the linear regression
    :type df: pandas.DataFrame
    :param metrics: List of metrics to be used for the linear regression (X)
    :type metrics: list of str
    :param label: Label to be used for the linear regression (y)
    :type label: str
    """
    st = time.time()

    X_train, X_test, y_train, y_test = train_test_split(df[metrics], df[label], test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    string_head = 'Linear Regression - Predict {}'.format(label)
    string_mse = 'Mean Squared Error: {}'.format(locale.format_string('%.2f', mse, grouping=True))
    string_inter = 'Intercept: {}'.format(locale.format_string('%.2f', model.intercept_, grouping=True))
    string_time = 'Execution time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - st)))

    [print(x) for x in [string_head, string_mse, string_inter, string_time, '']]
    with open('PredictPopularity/plots/predict/LinearRegression_Results.txt', 'a') as f:
        [f.write(x + '\n') for x in [string_head, string_mse, string_inter, string_time, '']]


# ---------------------------- Complex Regression ---------------------------- #

def complex_regression(df, metrics, label):
    """
    Perform complex regression on the given dataframe and save the results in a txt file

    :param df: Dataframe to be used for the complex regression
    :type df: pandas.DataFrame
    :param metrics: List of metrics to be used for the complex regression (X)
    :type metrics: list of str
    :param label: Label to be used for the complex regression (y)
    :type label: str
    """
    st = time.time()

    # only take 5000 rows - faster computation
    X_train, X_test, y_train, y_test = train_test_split(df.head(5000)[metrics], df.head(5000)[label], test_size=0.2,
                                                        random_state=42)

    model = PolynomialFeatures(degree=5)
    X_train_poly = model.fit_transform(X_train)
    X_test_poly = model.transform(X_test)
    regression = LinearRegression()
    regression.fit(X_train_poly, y_train)
    y_pred = regression.predict(X_test_poly)

    mse = mean_squared_error(y_test, y_pred)

    string_head = 'Complex Regression - Predict {}'.format(label)
    string_mse = 'Mean Squared Error: {}'.format(locale.format_string('%.2f', mse, grouping=True))
    string_inter = 'Intercept: {}'.format(locale.format_string('%.2f', regression.intercept_, grouping=True))
    string_time = 'Execution time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - st)))

    [print(x) for x in [string_head, string_mse, string_inter, string_time, '']]
    with open('PredictPopularity/plots/predict/ComplexRegression_Results.txt', 'a') as f:
        [f.write(x + '\n') for x in [string_head, string_mse, string_inter, string_time, '']]


# ---------------------------------------------------------------------------- #
#                    Calculate Popularity for Classification                   #
# ---------------------------------------------------------------------------- #

POPULARITY_LABELS = ['Low Popularity', 'Moderate Popularity', 'Good Popularity', 'High Popularity', 'Very High Popularity']


def calculate_popularity(df):
    """
    Calculate the popularity score and class for each song in the given dataframe
    Using the following metrics: Views, Streams, Likes, Comments

    :param df: Dataframe to be used for computing the popularity score and class
    :type df: pandas.DataFrame
    :return: new Dataframe including the popularity score and class for each song
    :rtype: pandas.DataFrame
    """
    # Calculate normalized popularity score
    normalized_likes = (df['Likes'] - df['Likes'].min()) / (df['Likes'].max() - df['Likes'].min())
    normalized_views = (df['Views'] - df['Views'].min()) / (df['Views'].max() - df['Views'].min())
    normalized_comments = (df['Comments'] - df['Comments'].min()) / (df['Comments'].max() - df['Comments'].min())
    normalized_streams = (df['Stream'] - df['Stream'].min()) / (df['Stream'].max() - df['Stream'].min())

    # Assign weights to normalize and calculate popularity score
    popularity_score = (normalized_views * 0.3) + (normalized_streams * 0.3) + (normalized_likes * 0.2) + (normalized_comments * 0.2)

    # Define popularity class thresholds
    popularity_thresholds = np.percentile(popularity_score, [0, 30, 50, 80, 90])

    # Assign popularity class based on popularity score
    popularity = np.select([popularity_score <= popularity_thresholds[1], popularity_score <= popularity_thresholds[2],
                            popularity_score <= popularity_thresholds[3], popularity_score <= popularity_thresholds[4],
                            popularity_score > popularity_thresholds[4]], POPULARITY_LABELS, default=POPULARITY_LABELS[0])

    df['Popularity Score'] = popularity_score
    df['Popularity'] = popularity
    return df


# ---------------------------------------------------------------------------- #
#               Try to predict/classify the popularity of a song               #
# ---------------------------------------------------------------------------- #

# Define train, validation, and test sizes
TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO = 0.7, 0.15, 0.15


# ---------------------------- Logistic Regression --------------------------- #

def logistic_regression(df, metrics, label, gridsearch=False):
    """
    Perform logistic regression on the given dataframe, save the results in a txt file and evaluate the model (plots)

    :param df: Dataframe to be used for the logistic regression
    :type df: pandas.DataFrame
    :param metrics: List of metrics to be used for the logistic regression (X)
    :type metrics: list of str
    :param label: Label to be used for the logistic regression (y)
    :type label: str
    :param gridsearch: Whether to perform gridsearch or not to find the best parameters for the model
    :type gridsearch: bool
    :return: accuracy of the model
    :rtype: float
    """
    st = time.time()

    X_train, X_test, y_train, y_test = train_test_split(df[metrics], df[label], test_size=1 - TRAIN_RATIO, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=TEST_RATIO / (TEST_RATIO + VALIDATION_RATIO), random_state=42)

    if gridsearch:
        parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
        gridsearch = GridSearchCV(LogisticRegression(max_iter=len(X_train)), parameters, scoring='accuracy')
        gridsearch.fit(X_valid, y_valid)
        string_params = 'Best parameters: {}'.format(gridsearch.best_params_)
        model = LogisticRegression(max_iter=len(X_train), C=gridsearch.best_params_['C'])
    else:
        model = LogisticRegression(max_iter=len(X_train))

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    string_head = 'Logistic Regression - Predict {}'.format(label)
    string_acc = 'Accuracy: {:.4f} ~ {:.2%}'.format(accuracy, accuracy)
    string_time = 'Execution time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - st)))

    output = [string_head, string_params, string_acc, string_time, ''] if gridsearch else [string_head, string_acc, string_time, '']
    [print(x) for x in output]
    with open('PredictPopularity/plots/classify/LogisticRegression_Results.txt', 'w') as f:
        [f.write(x + '\n') for x in output]

    evaluation_confusion_matrix(y_test, y_pred, model.predict_proba(X_test), 'LogisticRegression')

    return accuracy


# ------------------------- K-Nearest-Neighbors (KNN) ------------------------ #

def knn(df, metrics, label, gridsearch=False):
    """
    Perform KNN on the given dataframe, save the results in a txt file and evaluate the model (plots)

    :param df: Dataframe to be used for the KNN
    :type df: pandas.DataFrame
    :param metrics: List of metrics to be used for the KNN (X)
    :type metrics: list of str
    :param label: Label to be used for the KNN (y)
    :type label: str
    :param gridsearch: Whether to perform gridsearch or not to find the best parameters for the model
    :type gridsearch: bool
    :return: accuracy of the model
    :rtype: float
    """
    st = time.time()

    X_train, X_test, y_train, y_test = train_test_split(df[metrics], df[label], test_size=1 - TRAIN_RATIO,  random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=TEST_RATIO / (TEST_RATIO + VALIDATION_RATIO), random_state=42)

    if gridsearch:
        parameters = {"n_neighbors": range(1, 50), "weights": ["uniform", "distance"]}
        gridsearch = GridSearchCV(KNeighborsClassifier(), parameters, scoring='accuracy')
        gridsearch.fit(X_valid, y_valid)
        string_params = 'Best parameters: {}'.format(gridsearch.best_params_)
        model = KNeighborsClassifier(n_neighbors=gridsearch.best_params_["n_neighbors"], weights=gridsearch.best_params_["weights"])
    else:
        model = KNeighborsClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    string_head = 'KNN - Predict {}'.format(label)
    string_acc = 'Accuracy: {:.4f} ~ {:.2%}'.format(accuracy, accuracy)
    string_time = 'Execution time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - st)))

    output = [string_head, string_params, string_acc, string_time, ''] if gridsearch else [string_head, string_acc, string_time, '']
    [print(x) for x in output]
    with open('PredictPopularity/plots/classify/KNN_Results.txt', 'w') as f:
        [f.write(x + '\n') for x in output]

    evaluation_confusion_matrix(y_test, y_pred, model.predict_proba(X_test), 'KNN')

    return accuracy


# ----------------------- Support Vector Machine (SVM) ----------------------- #

def svm(df, metrics, label, gridsearch=False):
    """
    Perform SVM on the given dataframe, save the results in a txt file and evaluate the model (plots)

    :param df: Dataframe to be used for the SVM
    :type df: pandas.DataFrame
    :param metrics: List of metrics to be used for the SVM (X)
    :type metrics: list of str
    :param label: Label to be used for the SVM (y)
    :type label: str
    :param gridsearch: Whether to perform gridsearch or not to find the best parameters for the model
    :type gridsearch: bool
    :return: accuracy of the model
    :rtype: float
    """
    st = time.time()

    X_train, X_test, y_train, y_test = train_test_split(df[metrics], df[label], test_size=1 - TRAIN_RATIO, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=TEST_RATIO / (TEST_RATIO + VALIDATION_RATIO), random_state=42)

    if gridsearch:
        parameters = {'C': [0.1, 1, 5], 'kernel': ['rbf', 'poly', 'sigmoid']}
        gridsearch = GridSearchCV(SVC(), parameters, scoring='accuracy')
        gridsearch.fit(X_valid, y_valid)
        string_params = 'Best parameters: {}'.format(gridsearch.best_params_)
        model = SVC(probability=True, C=gridsearch.best_params_['C'], kernel=gridsearch.best_params_['kernel'])
    else:
        model = SVC(probability=True)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    string_head = 'SVM - Predict {}'.format(label)
    string_acc = 'Accuracy: {:.4f} ~ {:.2%}'.format(accuracy, accuracy)
    string_time = 'Execution time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - st)))

    output = [string_head, string_params, string_acc, string_time, ''] if gridsearch else [string_head, string_acc, string_time, '']
    [print(x) for x in output]
    with open('PredictPopularity/plots/classify/SVM_Results.txt', 'w') as f:
        [f.write(x + '\n') for x in output]

    evaluation_confusion_matrix(y_test, y_pred, model.predict_proba(X_test), 'SVM')

    return accuracy


# -------------------------------- Naive Bayes ------------------------------- #

def naive_bayes(df, metrics, label, gridsearch=False):
    """
    Perform Naive Bayes on the given dataframe, save the results in a txt file and evaluate the model (plots)

    :param df: Dataframe to be used for the Naive Bayes
    :type df: pandas.DataFrame
    :param metrics: List of metrics to be used for the Naive Bayes (X)
    :type metrics: list of str
    :param label: Label to be used for the Naive Bayes (y)
    :type label: str
    :param gridsearch: Whether to perform gridsearch or not to find the best parameters for the model
    :type gridsearch: bool
    :return: accuracy of the model
    :rtype: float
    """
    st = time.time()

    X_train, X_test, y_train, y_test = train_test_split(df[metrics], df[label], test_size=1 - TRAIN_RATIO, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=TEST_RATIO / (TEST_RATIO + VALIDATION_RATIO), random_state=42)

    if gridsearch:
        parameters = {'var_smoothing': np.logspace(0, -9, num=100)}
        gridsearch = GridSearchCV(GaussianNB(), parameters, scoring='accuracy')
        gridsearch.fit(X_valid, y_valid)
        string_params = 'Best parameters: {}'.format(gridsearch.best_params_)
        model = GaussianNB(var_smoothing=gridsearch.best_params_['var_smoothing'])
    else:
        model = GaussianNB()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    string_head = 'Naive Bayes - Predict {}'.format(label)
    string_acc = 'Accuracy: {:.4f} ~ {:.2%}'.format(accuracy, accuracy)
    string_time = 'Execution time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - st)))

    output = [string_head, string_params, string_acc, string_time, ''] if gridsearch else [string_head, string_acc, string_time, '']
    [print(x) for x in output]
    with open('PredictPopularity/plots/classify/NaiveBayes_Results.txt', 'w') as f:
        [f.write(x + '\n') for x in output]

    evaluation_confusion_matrix(y_test, y_pred, model.predict_proba(X_test), 'NaiveBayes')

    return accuracy


# ------------------------------- Decision Tree ------------------------------ #

def decision_tree(df, metrics, label, gridsearch=False):
    """
    Perform Decision Tree on the given dataframe, save the results in a txt file and evaluate the model (plots)

    :param df: Dataframe to be used for the Decision Tree
    :type df: pandas.DataFrame
    :param metrics: List of metrics to be used for the Decision Tree (X)
    :type metrics: list of str
    :param label: Label to be used for the Decision Tree (y)
    :type label: str
    :param gridsearch: Whether to perform gridsearch or not to find the best parameters for the model
    :type gridsearch: bool
    :return: accuracy of the model
    :rtype: float
    """
    st = time.time()

    X_train, X_test, y_train, y_test = train_test_split(df[metrics], df[label], test_size=1 - TRAIN_RATIO, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=TEST_RATIO / (TEST_RATIO + VALIDATION_RATIO), random_state=42)

    if gridsearch:
        parameters = {'max_features': ['sqrt', 'log2'], 'max_depth': [5, 6, 7, 8, 9], 'criterion': ['gini', 'entropy']}
        gridsearch = GridSearchCV(DecisionTreeClassifier(), parameters, scoring='accuracy')
        gridsearch.fit(X_valid, y_valid)
        string_params = 'Best parameters: {}'.format(gridsearch.best_params_)
        model = DecisionTreeClassifier(max_features=gridsearch.best_params_['max_features'], max_depth=gridsearch.best_params_['max_depth'], criterion=gridsearch.best_params_['criterion'])
    else:
        model = DecisionTreeClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    string_head = 'Decision Tree - Predict {}'.format(label)
    string_acc = 'Accuracy: {:.4f} ~ {:.2%}'.format(accuracy, accuracy)
    string_time = 'Execution time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - st)))

    output = [string_head, string_params, string_acc, string_time, ''] if gridsearch else [string_head, string_acc, string_time, '']
    [print(x) for x in output]
    with open('PredictPopularity/plots/classify/DecisionTree_Results.txt', 'w') as f:
        [f.write(x + '\n') for x in output]

    evaluation_confusion_matrix(y_test, y_pred, model.predict_proba(X_test), 'DecisionTree')

    return accuracy


# ------------------------------- Random Forest ------------------------------ #

def random_forest(df, metrics, label, gridsearch=False):
    """
    Perform Random Forest on the given dataframe, save the results in a txt file and evaluate the model (plots)

    :param df: Dataframe to be used for the Random Forest
    :type df: pandas.DataFrame
    :param metrics: List of metrics to be used for the Random Forest (X)
    :type metrics: list of str
    :param label: Label to be used for the Random Forest (y)
    :type label: str
    :param gridsearch: Whether to perform gridsearch or not to find the best parameters for the model
    :type gridsearch: bool
    :return: accuracy of the model
    :rtype: float
    """
    st = time.time()

    X_train, X_test, y_train, y_test = train_test_split(df[metrics], df[label], test_size=1 - TRAIN_RATIO, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=TEST_RATIO / (TEST_RATIO + VALIDATION_RATIO), random_state=42)

    if gridsearch:
        parameters = {'n_estimators': [100, 200], 'max_features': ['sqrt', 'log2'], 'max_depth': [3, 5, 8], 'criterion': ['gini', 'entropy']}
        gridsearch = GridSearchCV(RandomForestClassifier(), parameters, scoring='accuracy')
        gridsearch.fit(X_valid, y_valid)
        string_params = 'Best parameters: {}'.format(gridsearch.best_params_)
        model = RandomForestClassifier(n_estimators=gridsearch.best_params_['n_estimators'], max_features=gridsearch.best_params_['max_features'], max_depth=gridsearch.best_params_['max_depth'], criterion=gridsearch.best_params_['criterion'])
    else:
        model = RandomForestClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    string_head = 'Random Forest - Predict {}'.format(label)
    string_acc = 'Accuracy: {:.4f} ~ {:.2%}'.format(accuracy, accuracy)
    string_time = 'Execution time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - st)))

    output = [string_head, string_params, string_acc, string_time, ''] if gridsearch else [string_head, string_acc, string_time, '']
    [print(x) for x in output]
    with open('PredictPopularity/plots/classify/RandomForest_Results.txt', 'w') as f:
        [f.write(x + '\n') for x in output]

    evaluation_confusion_matrix(y_test, y_pred, model.predict_proba(X_test), 'RandomForest')

    return accuracy


# ----------------------------- Gradient Boosting ---------------------------- #

def gradient_boosting(df, metrics, label, gridsearch=False):
    """
    Perform Gradient Boosting on the given dataframe, save the results in a txt file and evaluate the model (plots)

    :param df: Dataframe to be used for the Gradient Boosting
    :type df: pandas.DataFrame
    :param metrics: List of metrics to be used for the Gradient Boosting (X)
    :type metrics: list of str
    :param label: Label to be used for the Gradient Boosting (y)
    :type label: str
    :param gridsearch: Whether to perform gridsearch or not to find the best parameters for the model
    :type gridsearch: bool
    :return: accuracy of the model
    :rtype: float
    """
    st = time.time()

    X_train, X_test, y_train, y_test = train_test_split(df[metrics], df[label], test_size=1 - TRAIN_RATIO, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=TEST_RATIO / (TEST_RATIO + VALIDATION_RATIO), random_state=42)

    if gridsearch:
        parameters = {"learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 8], "max_features": ["log2", "sqrt"]}
        gridsearch = GridSearchCV(GradientBoostingClassifier(), parameters, scoring='accuracy')
        gridsearch.fit(X_valid, y_valid)
        string_params = 'Best parameters: {}'.format(gridsearch.best_params_)
        model = GradientBoostingClassifier(learning_rate=gridsearch.best_params_['learning_rate'], max_depth=gridsearch.best_params_['max_depth'], max_features=gridsearch.best_params_['max_features'])
    else:
        model = GradientBoostingClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    string_head = 'Gradient Boosting - Predict {}'.format(label)
    string_acc = 'Accuracy: {:.4f} ~ {:.2%}'.format(accuracy, accuracy)
    string_time = 'Execution time: {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - st)))

    output = [string_head, string_params, string_acc, string_time, ''] if gridsearch else [string_head, string_acc, string_time, '']
    [print(x) for x in output]
    with open('PredictPopularity/plots/classify/GradientBoosting_Results.txt', 'w') as f:
        [f.write(x + '\n') for x in output]

    evaluation_confusion_matrix(y_test, y_pred, model.predict_proba(X_test), 'GradientBoosting')

    return accuracy


# ---------------------------------------------------------------------------- #
#                                  Evaluation                                  #
# ---------------------------------------------------------------------------- #

# ----------- Confusion Matrix (TP, FP, FN, TN) - Precision, Recall ---------- #

def evaluation_confusion_matrix(y_actual, y_pred, y_probas, name):
    """
    Plot the confusion matrix, precision recall table and ROC curves for the given model and save them in png files

    :param y_actual: Actual labels
    :type y_actual: list of str
    :param y_pred: Predicted labels
    :type y_pred: list of str
    :param y_probas: Predicted probabilities
    :type y_probas: Any
    :param name: Name of the model
    :type name: str
    """
    cm = confusion_matrix(y_actual, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=POPULARITY_LABELS).plot(cmap=plt.cm.YlOrRd, xticks_rotation=45)
    plt.tight_layout()
    plt.title('Confusion Matrix - {}'.format(name))
    plt.savefig('PredictPopularity/plots/classify/{}_ConfusionMatrix.png'.format(name), bbox_inches='tight')

    TP = np.diag(cm)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    data = {'TP': TP.astype(int), 'TN': TN.astype(int), 'FP': FP.astype(int), 'FN': FN.astype(int),
            'TPR': np.round(TP / (TP + FN), decimals=4),
            'TNR': np.round(TN / (TN + FP), decimals=4),
            'FPR': np.round(FP / (FP + TN), decimals=4),
            'FNR': np.round(FN / (TP + FN), decimals=4),
            'Precision': np.round(TP / (TP + FP), decimals=4),
            'Recall': np.round(TP / (TP + FN), decimals=4),
            'F1': np.round(2 * (TP / (TP + FP)) * (TP / (TP + FN)) / ((TP / (TP + FP)) + (TP / (TP + FN))), decimals=4)}
    df = pd.DataFrame(data=data, index=POPULARITY_LABELS)
    fig, ax = plt.subplots(figsize=(16, 2))
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')
    plt.title('Precision Recall Table - {}'.format(name))
    plt.savefig('PredictPopularity/plots/classify/{}_PrecisionRecall_Table.png'.format(name), bbox_inches='tight', dpi=300)

    skplt.metrics.plot_roc(y_actual, y_probas)
    plt.title('ROC Curves - {}'.format(name))
    plt.savefig('PredictPopularity/plots/classify/{}_ROC_Curves.png'.format(name))


if __name__ == '__main__':
    dataframe = read_data('data/Spotify_Youtube.csv')
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()

    correlation_heatmap(dataframe)

    linear_regression(dataframe, [x for x in numeric_columns if x != 'Views'], 'Views')
    linear_regression(dataframe, [x for x in numeric_columns if x != 'Likes'], 'Likes')
    linear_regression(dataframe, [x for x in numeric_columns if x != 'Comments'], 'Comments')
    linear_regression(dataframe, [x for x in numeric_columns if x != 'Stream'], 'Stream')

    complex_regression(dataframe, [x for x in numeric_columns if x != 'Views'], 'Views')
    complex_regression(dataframe, [x for x in numeric_columns if x != 'Likes'], 'Likes')
    complex_regression(dataframe, [x for x in numeric_columns if x != 'Comments'], 'Comments')
    complex_regression(dataframe, [x for x in numeric_columns if x != 'Stream'], 'Stream')

    dataframe = calculate_popularity(dataframe)
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()

    X = [x for x in numeric_columns if x != 'Popularity Score' and x != 'Popularity']
    y = 'Popularity'
    accuracies = {
        'Logistic Regression': logistic_regression(dataframe, X, y, gridsearch=True),
        'KNN': knn(dataframe, X, y, gridsearch=True),
        'SVM': svm(dataframe, X, y, gridsearch=True),
        'Naive Bayes': naive_bayes(dataframe, X, y, gridsearch=True),
        'Decision Tree': decision_tree(dataframe, X, y, gridsearch=True),
        'Random Forest': random_forest(dataframe, X, y, gridsearch=True),
        'Gradient Boosting': gradient_boosting(dataframe, X, y, gridsearch=True)}
    # accuracies = {
    #     'Logistic Regression': logistic_regression(dataframe, X, y),
    #     'KNN': knn(dataframe, X, y),
    #     'SVM': svm(dataframe, X, y),
    #     'Naive Bayes': naive_bayes(dataframe, X, y),
    #     'Decision Tree': decision_tree(dataframe, X, y),
    #     'Random Forest': random_forest(dataframe, X, y),
    #     'Gradient Boosting': gradient_boosting(dataframe, X, y)}

    print('--> Model with the highest accuracy = {}'.format(max(accuracies, key=accuracies.get)))

exit(0)
