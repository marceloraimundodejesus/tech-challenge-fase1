import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict

from .config import COLUNAS, NOMES_PT, PASTA_FIGURAS, ROTULOS_CLASSE, SEMENTE
from .modelagem import dobras


def _salvar(figura, nome):
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    figura.savefig(PASTA_FIGURAS / nome, dpi=120, bbox_inches="tight")


def escolher_limiar(modelo, x_treino, y_treino, recall_minimo=0.80):
    probabilidades = cross_val_predict(modelo, x_treino, y_treino, cv=dobras(), method="predict_proba")[:, 1]
    precisao, recall, limiares = precision_recall_curve(y_treino, probabilidades)
    precisao, recall = precisao[:-1], recall[:-1]
    candidatos = np.where(recall >= recall_minimo)[0]
    if len(candidatos) == 0:
        return 0.5, probabilidades
    melhor = candidatos[np.argmax(precisao[candidatos])]
    return float(limiares[melhor]), probabilidades


def grafico_precisao_recall_por_limiar(y_treino, probabilidades, limiar, nome_modelo, salvar=True):
    precisao, recall, limiares = precision_recall_curve(y_treino, probabilidades)
    figura, eixo = plt.subplots(figsize=(8, 5))
    eixo.plot(limiares, precisao[:-1], label="Precisão")
    eixo.plot(limiares, recall[:-1], label="Recall")
    eixo.axvline(0.5, color="gray", linestyle="--", label="Limiar padrão 0,5")
    eixo.axvline(limiar, color="red", linestyle="--", label=f"Limiar escolhido {limiar:.2f}")
    eixo.set_xlabel("Limiar de decisão")
    eixo.set_ylabel("Valor")
    eixo.set_title(f"Precisão e recall por limiar na validação cruzada ({nome_modelo})")
    eixo.legend()
    if salvar:
        _salvar(figura, "09_precisao_recall_por_limiar.png")
    return figura


def metricas_no_teste(modelos, x_teste, y_teste, limiares=None):
    limiares = limiares or {}
    linhas = []
    for nome, modelo in modelos.items():
        probabilidade = modelo.predict_proba(x_teste)[:, 1]
        limiar = limiares.get(nome, 0.5)
        previsto = (probabilidade >= limiar).astype(int)
        linhas.append({
            "modelo": nome,
            "limiar": limiar,
            "acuracia": accuracy_score(y_teste, previsto),
            "precisao": precision_score(y_teste, previsto, zero_division=0),
            "recall": recall_score(y_teste, previsto),
            "f1": f1_score(y_teste, previsto),
            "roc_auc": roc_auc_score(y_teste, probabilidade),
        })
    return pd.DataFrame(linhas).set_index("modelo").round(3)


def relatorio_classificacao(modelo, x_teste, y_teste, limiar=0.5):
    previsto = (modelo.predict_proba(x_teste)[:, 1] >= limiar).astype(int)
    return classification_report(y_teste, previsto, target_names=ROTULOS_CLASSE, digits=3)


def grafico_matrizes_confusao(modelos, x_teste, y_teste, limiares=None, salvar=True):
    limiares = limiares or {}
    figura, eixos = plt.subplots(1, len(modelos), figsize=(6 * len(modelos), 5))
    for eixo, (nome, modelo) in zip(np.atleast_1d(eixos), modelos.items()):
        limiar = limiares.get(nome, 0.5)
        previsto = (modelo.predict_proba(x_teste)[:, 1] >= limiar).astype(int)
        ConfusionMatrixDisplay.from_predictions(
            y_teste, previsto, display_labels=ROTULOS_CLASSE, cmap="Blues", ax=eixo, colorbar=False
        )
        eixo.set_title(f"{nome} (limiar {limiar:.2f})")
        eixo.set_xlabel("Previsto pelo modelo")
        eixo.set_ylabel("Real")
    figura.tight_layout()
    if salvar:
        _salvar(figura, "07_matrizes_confusao.png")
    return figura


def grafico_curva_roc(modelos, x_teste, y_teste, salvar=True):
    figura, eixo = plt.subplots(figsize=(7, 6))
    for nome, modelo in modelos.items():
        RocCurveDisplay.from_estimator(modelo, x_teste, y_teste, name=nome, ax=eixo)
    eixo.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chute aleatório (AUC 0,5)")
    eixo.set_xlabel("Taxa de falsos positivos")
    eixo.set_ylabel("Taxa de verdadeiros positivos (recall)")
    eixo.set_title("Curva ROC no conjunto de teste")
    eixo.legend(loc="lower right")
    if salvar:
        _salvar(figura, "08_curva_roc.png")
    return figura


def coeficientes_regressao_logistica(modelo):
    coeficientes = modelo.named_steps["modelo"].coef_[0]
    tabela = pd.DataFrame({"variavel": [NOMES_PT[c] for c in COLUNAS], "coeficiente": coeficientes})
    tabela["razao_de_chances"] = np.exp(tabela["coeficiente"])
    return tabela.sort_values("coeficiente", key=abs, ascending=False).set_index("variavel").round(3)


def importancias_random_forest(modelo, x_teste, y_teste):
    impureza = modelo.named_steps["modelo"].feature_importances_
    permutacao = permutation_importance(
        modelo, x_teste, y_teste, scoring="roc_auc", n_repeats=30, random_state=SEMENTE, n_jobs=-1
    )
    tabela = pd.DataFrame({
        "variavel": [NOMES_PT[c] for c in COLUNAS],
        "importancia_impureza": impureza,
        "importancia_permutacao": permutacao.importances_mean,
        "desvio_permutacao": permutacao.importances_std,
    })
    return tabela.sort_values("importancia_impureza", ascending=False).set_index("variavel").round(3)


def grafico_importancias(tabela_coeficientes, tabela_importancias, salvar=True):
    figura, eixos = plt.subplots(1, 2, figsize=(15, 5))
    coef = tabela_coeficientes["coeficiente"].sort_values()
    cores = ["#C44E52" if v > 0 else "#4C72B0" for v in coef.values]
    eixos[0].barh(coef.index, coef.values, color=cores)
    eixos[0].axvline(0, color="black", linewidth=0.8)
    eixos[0].set_title("Regressão Logística, peso de cada variável padronizada")
    eixos[0].set_xlabel("Coeficiente (positivo aumenta a chance de diabetes)")
    imp = tabela_importancias["importancia_impureza"].sort_values()
    eixos[1].barh(imp.index, imp.values, color="#55A868")
    eixos[1].set_title("Random Forest, importância das variáveis")
    eixos[1].set_xlabel("Importância (redução média de impureza)")
    figura.tight_layout()
    if salvar:
        _salvar(figura, "10_importancia_variaveis.png")
    return figura


def valores_shap_random_forest(modelo, x_teste):
    imputador = modelo.named_steps["imputador"]
    floresta = modelo.named_steps["modelo"]
    x_tratado = pd.DataFrame(imputador.transform(x_teste), columns=[NOMES_PT[c] for c in COLUNAS], index=x_teste.index)
    explicador = shap.TreeExplainer(floresta)
    valores = explicador(x_tratado)
    return valores[:, :, 1]


def grafico_shap_resumo(valores_shap, salvar=True):
    plt.figure()
    shap.plots.beeswarm(valores_shap, show=False, max_display=8)
    figura = plt.gcf()
    figura.set_size_inches(9, 5)
    plt.title("SHAP da Random Forest no teste, impacto na chance de diabetes")
    plt.xlabel("Valor SHAP (positivo empurra para Com diabetes)")
    if salvar:
        _salvar(figura, "11_shap_resumo.png")
    return figura


def grafico_shap_paciente(valores_shap, posicao, nome_arquivo, salvar=True):
    plt.figure()
    shap.plots.waterfall(valores_shap[posicao], show=False, max_display=8)
    figura = plt.gcf()
    figura.set_size_inches(9, 5)
    if salvar:
        _salvar(figura, nome_arquivo)
    return figura
