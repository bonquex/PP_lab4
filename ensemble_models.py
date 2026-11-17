"""Контрактная функция лабораторной №4: ансамбль для бинарной классификации."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

CREDIT_CATEGORICAL = [
    'Account_Balance', 'Payment_Status_of_Previous_Credit', 'Purpose',
    'Value_Savings_Stocks', 'Length_of_current_employment', 'Sex_Marital_Status',
    'Guarantors', 'Most_valuable_available_asset', 'Concurrent_Credits',
    'Type_of_apartment', 'Occupation', 'Telephone', 'Foreign_Worker',
]


def build_preprocessor(features, scale_numeric=False):
    """Все статистики и категории будут выучены внутри Pipeline.fit на train."""
    from sklearn.preprocessing import StandardScaler
    if isinstance(features, pd.DataFrame):
        categorical = [
            column for column in features.columns
            if column in CREDIT_CATEGORICAL
            or not pd.api.types.is_numeric_dtype(features[column])
        ]
        numeric = [column for column in features.columns if column not in categorical]
        steps = []
        if numeric:
            numeric_steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True))]
            if scale_numeric:
                numeric_steps.append(('scale', StandardScaler()))
            steps.append(('numeric', Pipeline(numeric_steps), numeric))
        if categorical:
            steps.append(('categorical', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]), categorical))
        return ColumnTransformer(steps)
    numeric_steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True))]
    if scale_numeric:
        numeric_steps.append(('scale', StandardScaler()))
    return Pipeline(numeric_steps)


# Конфигурация фиксируется по CV только на train из german.csv.
SELECTED_EXPERIMENT = 'rf_regularized'
MODEL_NAME = 'Random Forest (250 trees, max_depth=8, min_samples_leaf=3)'


def calculate_metrics(target, predictions, probabilities):
    """Положительный класс — 1, ROC-AUC считается по вероятностям."""
    target = np.asarray(target).reshape(-1)
    return {
        'roc_auc': float(roc_auc_score(target, probabilities)) if np.unique(target).size == 2 else float('nan'),
        'accuracy': float(accuracy_score(target, predictions)),
        'precision': float(precision_score(target, predictions, zero_division=0)),
        'recall': float(recall_score(target, predictions, zero_division=0)),
        'f1': float(f1_score(target, predictions, zero_division=0)),
    }


def fit_selected_model(features, target, random_seed=42):
    target = np.asarray(target).reshape(-1)
    if not np.array_equal(np.unique(target), [0, 1]):
        raise ValueError('В обучающей выборке должны быть оба класса: 0 и 1')
    model = Pipeline([
        ('prepare', build_preprocessor(features)),
        ('model', RandomForestClassifier(
            n_estimators=250, max_depth=8, min_samples_leaf=3,
            max_features='sqrt', random_state=random_seed, n_jobs=1,
        )),
    ])
    model.fit(features, target)
    return model


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучаю ансамбль и предобработку по train; test_target нужен только для метрик."""
    model = fit_selected_model(data.train_features, data.train_target, data.random_seed)
    probabilities = model.predict_proba(data.test_features)[:, list(model.classes_).index(1)]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = calculate_metrics(data.test_target, predictions, probabilities)
    return EnsembleResult(
        model_name=MODEL_NAME,
        probabilities=probabilities,
        predictions=predictions,
        **metrics,
    )
