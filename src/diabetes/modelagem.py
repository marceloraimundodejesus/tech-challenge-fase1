import pandas as pd
from sklearn.metrics import make_scorer, precision_score
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DOBRAS, SEMENTE

METRICAS = {
    "acuracia": "accuracy",
    "precisao": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def dobras():
    return StratifiedKFold(n_splits=DOBRAS, shuffle=True, random_state=SEMENTE)


def pipeline_base():
    return Pipeline([
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("modelo", DummyClassifier(strategy="most_frequent")),
    ])


def pipeline_regressao_logistica():
    return Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escala", StandardScaler()),
        ("modelo", LogisticRegression(max_iter=1000, random_state=SEMENTE)),
    ])


def pipeline_random_forest():
    return Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("modelo", RandomForestClassifier(random_state=SEMENTE, n_jobs=-1)),
    ])


GRADE_REGRESSAO_LOGISTICA = {
    "modelo__C": [0.01, 0.1, 1, 10],
    "modelo__class_weight": [None, "balanced"],
}

GRADE_RANDOM_FOREST = {
    "modelo__n_estimators": [200, 400],
    "modelo__max_depth": [4, 6, 8, None],
    "modelo__min_samples_leaf": [1, 5, 10],
    "modelo__class_weight": [None, "balanced"],
}


def ajustar_hiperparametros(pipeline, grade, x_treino, y_treino, metrica="f1"):
    busca = GridSearchCV(pipeline, grade, scoring=metrica, cv=dobras(), n_jobs=-1, refit=True)
    busca.fit(x_treino, y_treino)
    return busca


def comparar_na_validacao_cruzada(modelos, x_treino, y_treino):
    linhas = []
    for nome, modelo in modelos.items():
        resultado = cross_validate(modelo, x_treino, y_treino, cv=dobras(), scoring=METRICAS, n_jobs=-1)
        linha = {"modelo": nome}
        for metrica in METRICAS:
            valores = resultado[f"test_{metrica}"]
            linha[f"{metrica}_media"] = valores.mean()
            linha[f"{metrica}_desvio"] = valores.std()
        linhas.append(linha)
    return pd.DataFrame(linhas).set_index("modelo").round(3)
