from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, fbeta_score, roc_curve, auc, confusion_matrix 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import VotingClassifier
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_status_distribution(status_counts):
    """
    Plot the distribution of payment statuses over time.
    
    Args:
    - status_counts (Series): Counts of payment statuses.
    
    Returns:
    - None
    """
    plt.figure(figsize=(14, 10))
    status_counts.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='tab10')
    plt.title('Payment Status Distribution Over Time')
    plt.xlabel('Months Balance (0 = Current Month)')
    plt.ylabel('Number of Records')
    plt.legend(title='Payment Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def classify_customer(payment_history):
    """
    Classify a customer based on their payment history.
    
    Args:
    - payment_history (Series): Series of payment statuses.
    
    Returns:
    - str: 'bad', 'good', or 'intermediate' based on payment history.
    """
    recent_6_months = payment_history.head(6)
    if any(recent_6_months.isin(['2', '3', '4', '5'])):
        return 'bad'
    elif all(recent_6_months.isin(['C', '0', '1'])):
        return 'good'
    else:
        return 'intermediate'


def refine_intermediate_classification_v3(payment_history):
    """
    Refine classification for intermediate customers based on their payment history.
    
    Args:
    - payment_history (Series): Series of payment statuses.
    
    Returns:
    - str: 'bad' or 'good' based on refined criteria.
    """
    recent_6_months = payment_history.head(9)
    if recent_6_months[recent_6_months.isin(['1'])].count() >= 2:
        return 'bad'
    return 'good'


def count_nan_values(df):
    """
    Count NaN values in a DataFrame.
    
    Args:
    - df (DataFrame): Dataframe to check for NaN values.
    
    Returns:
    - Series: Columns with NaN counts greater than 0.
    """
    nan_counts = df.isna().sum()
    return nan_counts[nan_counts > 0]


def filter_nan_years(df):
    """
    Filter rows where YEARS_EMPLOYED is NaN and display associated NAME_INCOME_TYPE values.
    
    Args:
    - df (DataFrame): Dataframe to filter.
    
    Returns:
    - Series: Value counts of NAME_INCOME_TYPE for rows where YEARS_EMPLOYED is NaN.
    """
    return df[df['YEARS_EMPLOYED'].isna()]['NAME_INCOME_TYPE'].value_counts()

def plot_count_distribution(data, column, hue=None, ax=None, palette='Set2', title="", xlabels=[]):
    """
    Plot a count distribution for a given column.
    
    Args:
    - data (DataFrame): Data to be plotted.
    - column (str): Column name to plot distribution for.
    - hue (str, optional): Column in `data` to use for hue.
    - ax (AxesSubplot, optional): Matplotlib axis object.
    - palette (str, optional): Color palette for plotting.
    - title (str, optional): Title for the plot.
    - xlabels (list, optional): Labels for x-axis ticks.
    
    Returns:
    - None
    """
    if ax is None:
        fig, ax = plt.subplots()
    sns.countplot(data=data, x=column, hue=hue, palette=palette, ax=ax)
    ax.set_title(title)
    if xlabels:
        ax.set_xticklabels(xlabels)
    ax.set_xlabel(column)
    ax.set_ylabel('Count')


def plot_histogram(data, column, ax, color, title):
    """
    Plot a histogram for a given column.
    
    Args:
    - data (DataFrame): Data to be plotted.
    - column (str): Column name for which histogram is to be plotted.
    - ax (AxesSubplot): Matplotlib axis object.
    - color (str): Color for the histogram bars.
    - title (str): Title for the histogram.
    
    Returns:
    - None
    """
    sns.histplot(data[column], kde=True, ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel('Count')


def evaluate_model(y_true, y_pred):
    """
    Evaluate the performance of a classification model using various metrics.
    
    Args:
    - y_true (Series or array): True labels.
    - y_pred (Series or array): Predicted labels by the model.
    
    Returns:
    - dict: Dictionary containing various evaluation metrics.
    """
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'F2-Score': fbeta_score(y_true, y_pred, beta=2)
    }


def plot_custom_confusion_matrix(ax, y_true, y_pred):
    """
    Plot a custom confusion matrix.
    
    Args:
    - ax (AxesSubplot): Matplotlib axis object.
    - y_true (Series or array): True labels.
    - y_pred (Series or array): Predicted labels by the model.
    
    Returns:
    - None
    """
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, fmt=".2f", 
                xticklabels=["No Risk", "Risk"], yticklabels=["No Risk", "Risk"], ax=ax)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

def plot_roc_auc(ax, y_true, y_pred_prob):
    """
    Plot the Receiver Operating Characteristic (ROC) curve along with the Area Under the Curve (AUC).
    
    Args:
    - ax (AxesSubplot): Matplotlib axis object.
    - y_true (Series or array): True labels.
    - y_pred_prob (Series or array): Predicted probabilities by the model for the positive class.
    
    Returns:
    - None
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")

def train_and_evaluate(models, strategies, X_train, y_train, X_val, y_val):
    """
    Train various models using different resampling strategies and evaluate their performance.
    
    Args:
    - models (dict): Dictionary with model names as keys and tuples of (model instance, parameter grid) as values.
    - strategies (dict): Dictionary with strategy names as keys and resampler instances as values.
    - X_train (DataFrame): Training features.
    - y_train (Series or array): Training labels.
    - X_val (DataFrame): Validation features.
    - y_val (Series or array): Validation labels.
    
    Returns:
    - tuple: (results, best_model_instance) where results is a dictionary with model and strategy names as keys and evaluation metrics as values, 
      and best_model_instance is the best model instance based on F1-score.
    """
    results = {}
    best_f1_score = -np.inf
    best_model_instance = None
    
    for model_name, (model, params) in models.items():
        for strategy_name, resampler in strategies.items():
            print(f"Training {model_name} with {strategy_name}...")            
            X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
            with warnings.catch_warnings(): 
                warnings.simplefilter("ignore")            
                random_search = RandomizedSearchCV(model, params, n_iter=10, cv=4, scoring='f1', n_jobs=-1, random_state=42)
                random_search.fit(X_resampled, y_resampled)
            best_model = random_search.best_estimator_
            print(best_model)
            y_val_pred = best_model.predict(X_val)
            metrics = evaluate_model(y_val, y_val_pred)
            results[(model_name, strategy_name)] = {
                'metrics': metrics,
                'model': best_model,
                'best_params': random_search.best_params_
            }  
            current_f1_score = metrics['F1-Score']
            if current_f1_score > best_f1_score:
                best_f1_score = current_f1_score
                best_model_instance = best_model
            # Visualization
            visualize_evaluation(X_val, y_val, y_val_pred, best_model, f"{model_name} with {strategy_name}")
    
    return results, best_model_instance

def visualize_evaluation(X, y_true, y_pred, model, title):
    """
    Visualize the performance of a trained model using a confusion matrix and ROC-AUC curve.
    
    Args:
    - X (DataFrame): Features.
    - y_true (Series or array): True labels.
    - y_pred (Series or array): Predicted labels by the model.
    - model (estimator): Trained model instance.
    - title (str): Title for the visualization.
    
    Returns:
    - None
    """
    y_prob = model.predict_proba(X)[:, 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    fig.suptitle(title, fontsize=10)
    
    # Confusion Matrix
    plot_custom_confusion_matrix(ax1, y_true, y_pred)
    
    # ROC-AUC
    plot_roc_auc(ax2, y_true, y_prob)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

def evaluate_final_model(model, X, y):
    """
    Evaluate a given model on provided data.
    
    Args:
    - model (estimator): Trained machine learning model.
    - X (DataFrame or array-like): Data features.
    - y (Series or array-like): True labels.

    Returns:
    - tuple: Tuple containing performance metrics, predicted labels, and predicted probabilities.
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    metrics = evaluate_model(y, y_pred)
    return metrics, y_pred, y_prob


def plot_evaluation(y_true, y_pred, y_prob):
    """
    Visualize the performance of a classifier using a confusion matrix and ROC-AUC curve.
    
    Args:
    - y_true (Series or array-like): True labels.
    - y_pred (Series or array-like): Predicted labels.
    - y_prob (array-like): Predicted probabilities for the positive class.

    Returns:
    - None
    """
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    
    # Confusion Matrix
    plot_custom_confusion_matrix(axes[0], y_true, y_pred)
    axes[0].set_title("Confusion Matrix on Test Set")
    
    # ROC-AUC
    plot_roc_auc(axes[1], y_true, y_prob)
    axes[1].set_title("ROC Curve on Test Set")
    
    plt.tight_layout()
    plt.show()


def evaluate_classifier(model, X, y):
    """
    Evaluate the performance metrics of a classifier.
    
    Args:
    - model (estimator): Trained machine learning model.
    - X (DataFrame or array-like): Data features.
    - y (Series or array-like): True labels.

    Returns:
    - dict: Dictionary containing various performance metrics.
    """
    y_pred = model.predict(X)
    metrics = {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred),
        'Recall': recall_score(y, y_pred),
        'F1 Score': f1_score(y, y_pred),
        'F2 Score': fbeta_score(y, y_pred, beta=2)
    }
    return metrics


def train_models(models_and_params, X, y):
    """
    Train machine learning models using RandomizedSearchCV for hyperparameter tuning.
    
    Args:
    - models_and_params (dict): Dictionary containing models and their hyperparameters.
    - X (DataFrame or array-like): Data features.
    - y (Series or array-like): True labels.

    Returns:
    - dict: Dictionary containing best models for each algorithm.
    """
    best_models = {}
    for model_name, model_info in models_and_params.items():
        print(f"Training {model_name}...")
        random_search = RandomizedSearchCV(
            model_info['model'], 
            model_info['params'], 
            n_iter=10, 
            cv=3, 
            scoring='f1', 
            n_jobs=-1, 
            random_state=42
        )
        random_search.fit(X, y)
        best_models[model_name] = random_search.best_estimator_
    return best_models


def evaluate_and_print(model, X, y, name):
    """
    Evaluate a classifier and print its performance metrics.
    
    Args:
    - model (estimator): Trained machine learning model.
    - X (DataFrame or array-like): Data features.
    - y (Series or array-like): True labels.
    - name (str): Name of the classifier/model.

    Returns:
    - None
    """
    metrics = evaluate_classifier(model, X, y)
    print(f"\nPerformance of {name} on Validation Set:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

def evaluate_and_visualize_ensemble(models, X, y):
    """
    Evaluate and visualize the performance of multiple models using confusion matrix and ROC-AUC curve.
    
    Args:
    - models (dict): Dictionary of trained models with model names as keys.
    - X (DataFrame): Features for evaluation.
    - y (Series or array): True labels.
    
    Returns:
    - DataFrame: A DataFrame with performance metrics for each model.
    """
    all_metrics = []
    for model_name, model in models.items():
        y_pred = model.predict(X)
        metrics_df = metrics_to_dataframe(y, y_pred, model_name)
        all_metrics.append(metrics_df)
        plot_evaluation_metrics(model, X, y, model_name)
    return pd.concat(all_metrics, axis=0)

def metrics_to_dataframe(y_true, y_pred, model_name):
    """
    Convert evaluation metrics to a DataFrame for a given model.
    
    Args:
    - y_true (Series or array): True labels.
    - y_pred (Series or array): Predicted labels by the model.
    - model_name (str): Name of the model.
    
    Returns:
    - DataFrame: A DataFrame with evaluation metrics.
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'F2-Score': fbeta_score(y_true, y_pred, beta=2)
    }
    return pd.DataFrame(metrics, index=[model_name])

def plot_evaluation_metrics(model, X, y, model_name):
    """
    Visualize the performance metrics of a model using confusion matrix and ROC-AUC curve.
    
    Args:
    - model (estimator): Trained model instance.
    - X (DataFrame): Features for evaluation.
    - y (Series or array): True labels.
    - model_name (str): Name of the model.
    
    Returns:
    - None
    """
    y_pred = model.predict(X)
    y_pred_prob = model.predict_proba(X)[:, 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    
    # Confusion Matrix
    sns.heatmap(confusion_matrix(y, y_pred), annot=True, cmap=plt.cm.Blues, fmt=".2f", 
                xticklabels=["No Risk", "Risk"], yticklabels=["No Risk", "Risk"], ax=ax1)
    ax1.set_title(f"Confusion Matrix for {model_name}", fontsize=10)
    
    # ROC-AUC
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')    
    ax2.set_title(f"ROC Curve for {model_name}", fontsize=10)
    ax2.legend(loc="lower right")
    
    plt.tight_layout()
    plt.show()

def train_models_with_resampling(models, resampling_strategies, X_train, y_train, X_val, y_val):
    """
    Train models with different resampling strategies and evaluate them on the validation set.
    
    Args:
    - models (dict): Dictionary containing model names as keys and (model, parameters) tuples as values.
    - resampling_strategies (dict): Dictionary of resampling strategies with strategy names as keys and strategy functions as values.
    - X_train (DataFrame): Training data features.
    - y_train (Series or array): Training data labels.
    - X_val (DataFrame): Validation data features.
    - y_val (Series or array): Validation data labels.
    
    Returns:
    - dict: Dictionary containing evaluation results for each model and strategy combination.
    - estimator: Best model instance based on F1-Score.
    """
    results={}
    best_f1_score = -np.inf
    best_model_instance = None

    for model_name, (model, params) in models.items():
        for strategy_name, resampler in resampling_strategies.items():
            print(f"Training {model_name} with {strategy_name}...")
            X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                random_search = RandomizedSearchCV(model, params, n_iter=10, cv=4, scoring='f1', n_jobs=-1, random_state=42)
                random_search.fit(X_resampled, y_resampled)

            best_model = random_search.best_estimator_
            print(best_model)
            y_val_pred = best_model.predict(X_val)
            metrics = evaluate_model(y_val, y_val_pred)
            
            results[(model_name, strategy_name)] = {
                'metrics': metrics,
                'model': best_model,
                'best_params': random_search.best_params_
            }

            # Check for best model
            current_f1_score = metrics['F1-Score']
            if current_f1_score > best_f1_score:
                best_f1_score = current_f1_score
                best_model_instance = best_model
            
            # Visualization
            visualize_evaluation(X_val, y_val, y_val_pred, best_model, f"{model_name} with {strategy_name}")


    return results, best_model_instance

def train_and_evaluate_ensemble(best_models, ensemble_configs, X_train, y_train, X_val, y_val):
    """
    Train ensemble models using pre-trained models and evaluate them on the validation set.
    
    Args:
    - best_models (dict): Dictionary of pre-trained models.
    - ensemble_configs (dict): Dictionary defining which models to combine for each ensemble.
    - X_train (DataFrame): Training data features.
    - y_train (Series or array): Training data labels.
    - X_val (DataFrame): Validation data features.
    - y_val (Series or array): Validation data labels.
    
    Returns:
    - dict: Dictionary of trained ensemble models.
    """
    trained_ensembles = {}
    for ensemble_name, model_names in ensemble_configs.items():
        # Assicurati di utilizzare i modelli addestrati con le caratteristiche selezionate
        estimators = [(name, best_models[name]) for name in model_names]
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble.fit(X_train, y_train)
        
        # Salva l'ensemble addestrato
        trained_ensembles[ensemble_name] = ensemble
        
        print(f"\nPerformance of {ensemble_name} Ensemble on Validation Set:")
        metrics = evaluate_classifier(ensemble, X_val, y_val)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    return trained_ensembles

def train_single_models(models_and_params, X_train, y_train):
    """
    Train multiple models on the training data using RandomizedSearchCV for hyperparameter tuning.
    
    Args:
    - models_and_params (dict): Dictionary containing model names as keys and (model, parameters) tuples as values.
    - X_train (DataFrame): Training data features.
    - y_train (Series or array): Training data labels.
    
    Returns:
    - dict: Dictionary of best models after hyperparameter tuning.
    """
    best_models = {}
    
    for model_name, (model, params) in models_and_params.items():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            print(f"Training {model_name}...")
            random_search = RandomizedSearchCV(
                model, params, n_iter=10, cv=3, scoring='f1', n_jobs=-1, random_state=42)
            random_search.fit(X_train, y_train)
            best_models[model_name] = random_search.best_estimator_

    return best_models


def evaluate_and_visualize_model(model, X_test, y_test, model_name):
    """
    Evaluate and visualize the performance of a model on the test set using metrics and plots.
    
    Args:
    - model (estimator): Trained model instance.
    - X_test (DataFrame): Test data features.
    - y_test (Series or array): Test data labels.
    - model_name (str): Name of the model.
    
    Returns:
    - DataFrame: A DataFrame with evaluation metrics.
    """
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics_df = metrics_to_dataframe(y_test, y_pred, model_name)
    
    # Visualize metrics
    plot_evaluation_metrics(model, X_test, y_test, model_name)
    
    return metrics_df


def find_best_match(feature_str, columns):
    """
    Find the best matching column name from a list of column names for a given feature string.
    
    Args:
    - feature_str (str): The feature string to match.
    - columns (list): List of column names.
    
    Returns:
    - str: The best matching column name.
    """
    best_match = None
    best_match_length = 0
    for col in columns:
        if col in feature_str and len(col) > best_match_length:
            best_match = col
            best_match_length = len(col)
    return best_match
