import matplotlib.pyplot as plt
import seaborn as sns

from .config import ALVO, COLUNAS, NOMES_PT, PASTA_FIGURAS, ROTULOS_CLASSE

CORES = {0: "#4C72B0", 1: "#DD8452"}


def _salvar(figura, nome):
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    figura.savefig(PASTA_FIGURAS / nome, dpi=120, bbox_inches="tight")


def grafico_distribuicao_alvo(dados, salvar=True):
    contagem = dados[ALVO].value_counts().sort_index()
    percentual = contagem / contagem.sum() * 100
    figura, eixo = plt.subplots(figsize=(6, 4))
    barras = eixo.bar(ROTULOS_CLASSE, contagem.values, color=[CORES[0], CORES[1]])
    for barra, qtd, pct in zip(barras, contagem.values, percentual.values):
        eixo.text(barra.get_x() + barra.get_width() / 2, qtd + 5, f"{qtd} ({pct:.1f}%)", ha="center")
    eixo.set_title("Quantas pacientes têm diabetes na base")
    eixo.set_ylabel("Pacientes")
    eixo.set_ylim(0, contagem.max() * 1.15)
    if salvar:
        _salvar(figura, "01_distribuicao_alvo.png")
    return figura


def grafico_zeros_invalidos(tabela_zeros, salvar=True):
    tabela = tabela_zeros.sort_values("percentual")
    figura, eixo = plt.subplots(figsize=(7, 4))
    eixo.barh([NOMES_PT[c] for c in tabela.index], tabela["percentual"], color="#C44E52")
    for i, (qtd, pct) in enumerate(zip(tabela["zeros"], tabela["percentual"])):
        eixo.text(pct + 0.5, i, f"{qtd} ({pct}%)", va="center")
    eixo.set_title("Zeros que na verdade são dados faltando")
    eixo.set_xlabel("Percentual das pacientes")
    eixo.set_xlim(0, tabela["percentual"].max() * 1.25)
    if salvar:
        _salvar(figura, "02_zeros_invalidos.png")
    return figura


def grafico_histogramas_por_classe(dados, salvar=True):
    figura, eixos = plt.subplots(2, 4, figsize=(16, 7))
    for eixo, coluna in zip(eixos.ravel(), COLUNAS):
        sns.histplot(data=dados, x=coluna, hue=ALVO, kde=True, ax=eixo, palette=CORES, stat="density", common_norm=False)
        eixo.set_title(NOMES_PT[coluna])
        eixo.set_xlabel("")
        eixo.set_ylabel("Densidade")
        legenda = eixo.get_legend()
        if legenda is not None:
            legenda.set_title("")
            for texto, rotulo in zip(legenda.texts, ROTULOS_CLASSE):
                texto.set_text(rotulo)
    figura.suptitle("Como cada variável se distribui em quem tem e em quem não tem diabetes", fontsize=14)
    figura.tight_layout()
    if salvar:
        _salvar(figura, "03_histogramas_por_classe.png")
    return figura


def grafico_boxplots_por_classe(dados, salvar=True):
    figura, eixos = plt.subplots(2, 4, figsize=(16, 7))
    base = dados.assign(Grupo=dados[ALVO].map({0: ROTULOS_CLASSE[0], 1: ROTULOS_CLASSE[1]}))
    paleta = {ROTULOS_CLASSE[0]: CORES[0], ROTULOS_CLASSE[1]: CORES[1]}
    for eixo, coluna in zip(eixos.ravel(), COLUNAS):
        sns.boxplot(data=base, x="Grupo", y=coluna, hue="Grupo", ax=eixo, palette=paleta, legend=False)
        eixo.set_title(NOMES_PT[coluna])
        eixo.set_xlabel("")
        eixo.set_ylabel("")
    figura.suptitle("Medianas e valores extremos por grupo", fontsize=14)
    figura.tight_layout()
    if salvar:
        _salvar(figura, "04_boxplots_por_classe.png")
    return figura


def grafico_correlacao(dados, salvar=True):
    correlacao = dados[COLUNAS + [ALVO]].corr(method="pearson")
    correlacao = correlacao.rename(index=NOMES_PT, columns=NOMES_PT)
    figura, eixo = plt.subplots(figsize=(9, 7))
    sns.heatmap(correlacao, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=eixo)
    eixo.set_title("Correlação de Pearson entre as variáveis")
    if salvar:
        _salvar(figura, "05_correlacao.png")
    return figura


def grafico_glicose_imc(dados, salvar=True):
    figura, eixo = plt.subplots(figsize=(8, 6))
    for classe, rotulo in enumerate(ROTULOS_CLASSE):
        parte = dados[dados[ALVO] == classe]
        eixo.scatter(parte["Glucose"], parte["BMI"], alpha=0.6, label=rotulo, color=CORES[classe])
    eixo.set_xlabel("Glicose (teste de 2 horas)")
    eixo.set_ylabel("IMC")
    eixo.set_title("Glicose e IMC juntas separam bem, mas não perfeitamente")
    eixo.legend()
    if salvar:
        _salvar(figura, "06_glicose_imc.png")
    return figura
