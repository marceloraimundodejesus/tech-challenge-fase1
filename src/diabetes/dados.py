import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ALVO, CAMINHO_DADOS, COLUNAS, COLUNAS_ZERO_INVALIDO, SEMENTE, TAMANHO_TESTE


def carregar_dados(caminho=CAMINHO_DADOS):
    dados = pd.read_csv(caminho)
    faltando = set(COLUNAS + [ALVO]) - set(dados.columns)
    if faltando:
        raise ValueError(f"Colunas que não encontrei no arquivo {sorted(faltando)}")
    return dados[COLUNAS + [ALVO]]


def contar_zeros_invalidos(dados):
    zeros = (dados[COLUNAS_ZERO_INVALIDO] == 0).sum()
    percentual = (zeros / len(dados) * 100).round(1)
    return pd.DataFrame({"zeros": zeros, "percentual": percentual})


def marcar_zeros_como_faltantes(dados):
    tratado = dados.copy()
    tratado[COLUNAS_ZERO_INVALIDO] = tratado[COLUNAS_ZERO_INVALIDO].replace(0, np.nan)
    return tratado


def separar_treino_teste(dados):
    x = dados[COLUNAS]
    y = dados[ALVO]
    return train_test_split(x, y, test_size=TAMANHO_TESTE, stratify=y, random_state=SEMENTE)
