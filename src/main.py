import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from diabetes import avaliacao, exploracao
from diabetes.config import PASTA_RESULTADOS
from diabetes.dados import carregar_dados, contar_zeros_invalidos, marcar_zeros_como_faltantes, separar_treino_teste
from diabetes.modelagem import (
    GRADE_RANDOM_FOREST,
    GRADE_REGRESSAO_LOGISTICA,
    ajustar_hiperparametros,
    comparar_na_validacao_cruzada,
    pipeline_base,
    pipeline_random_forest,
    pipeline_regressao_logistica,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def etapa(texto):
    print(f"\n{texto}")


def main():
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    etapa("Carregando a base")
    dados_brutos = carregar_dados()
    print(f"{len(dados_brutos)} pacientes e {dados_brutos.shape[1] - 1} variáveis")

    etapa("Zeros que viram dado faltando")
    tabela_zeros = contar_zeros_invalidos(dados_brutos)
    print(tabela_zeros.to_string())
    dados = marcar_zeros_como_faltantes(dados_brutos)

    etapa("Gerando os gráficos da análise exploratória")
    exploracao.grafico_distribuicao_alvo(dados_brutos)
    exploracao.grafico_zeros_invalidos(tabela_zeros)
    exploracao.grafico_histogramas_por_classe(dados)
    exploracao.grafico_boxplots_por_classe(dados)
    exploracao.grafico_correlacao(dados)
    exploracao.grafico_glicose_imc(dados)
    plt.close("all")

    etapa("Separando treino e teste estratificado")
    x_treino, x_teste, y_treino, y_teste = separar_treino_teste(dados)
    print(f"Treino {len(x_treino)} pacientes, teste {len(x_teste)} pacientes")

    etapa("Buscando os hiperparâmetros com validação cruzada (pode levar uns minutos)")
    busca_rl = ajustar_hiperparametros(pipeline_regressao_logistica(), GRADE_REGRESSAO_LOGISTICA, x_treino, y_treino)
    busca_rf = ajustar_hiperparametros(pipeline_random_forest(), GRADE_RANDOM_FOREST, x_treino, y_treino)
    print("Regressão Logística", busca_rl.best_params_)
    print("Random Forest", busca_rf.best_params_)

    modelos = {
        "Regressão Logística": busca_rl.best_estimator_,
        "Random Forest": busca_rf.best_estimator_,
    }

    etapa("Comparando na validação cruzada do treino (média e desvio de 5 dobras)")
    comparacao = comparar_na_validacao_cruzada({"Modelo base (Dummy)": pipeline_base(), **modelos}, x_treino, y_treino)
    print(comparacao.to_string())

    etapa("Escolhendo o limiar da Random Forest na validação cruzada para recall de pelo menos 80%")
    limiar_rf, probabilidades_oof = avaliacao.escolher_limiar(busca_rf.best_estimator_, x_treino, y_treino)
    avaliacao.grafico_precisao_recall_por_limiar(y_treino, probabilidades_oof, limiar_rf, "Random Forest")
    print(f"Limiar escolhido {limiar_rf:.3f}")

    etapa("Avaliando no teste, usado uma única vez")
    teste_padrao = avaliacao.metricas_no_teste(modelos, x_teste, y_teste)
    teste_ajustado = avaliacao.metricas_no_teste(
        {"Random Forest com limiar ajustado": busca_rf.best_estimator_}, x_teste, y_teste,
        {"Random Forest com limiar ajustado": limiar_rf},
    )
    print(teste_padrao.to_string())
    print(teste_ajustado.to_string())
    relatorios = {
        nome: avaliacao.relatorio_classificacao(modelo, x_teste, y_teste) for nome, modelo in modelos.items()
    }
    relatorios["Random Forest com limiar ajustado"] = avaliacao.relatorio_classificacao(
        busca_rf.best_estimator_, x_teste, y_teste, limiar_rf
    )
    for nome, texto in relatorios.items():
        print(f"\n{nome}\n{texto}")
    avaliacao.grafico_matrizes_confusao(modelos, x_teste, y_teste)
    avaliacao.grafico_curva_roc(modelos, x_teste, y_teste)
    plt.close("all")

    etapa("Interpretando os modelos")
    coeficientes = avaliacao.coeficientes_regressao_logistica(busca_rl.best_estimator_)
    importancias = avaliacao.importancias_random_forest(busca_rf.best_estimator_, x_teste, y_teste)
    print(coeficientes.to_string())
    print(importancias.to_string())
    avaliacao.grafico_importancias(coeficientes, importancias)
    valores_shap = avaliacao.valores_shap_random_forest(busca_rf.best_estimator_, x_teste)
    avaliacao.grafico_shap_resumo(valores_shap)
    probabilidades_teste = busca_rf.best_estimator_.predict_proba(x_teste)[:, 1]
    avaliacao.grafico_shap_paciente(valores_shap, int(probabilidades_teste.argmax()), "12_shap_paciente_maior_risco.png")
    avaliacao.grafico_shap_paciente(valores_shap, int(probabilidades_teste.argmin()), "13_shap_paciente_menor_risco.png")
    plt.close("all")

    etapa("Salvando os resultados")
    comparacao.to_csv(PASTA_RESULTADOS / "comparacao_validacao_cruzada.csv")
    teste_padrao.to_csv(PASTA_RESULTADOS / "metricas_teste.csv")
    teste_ajustado.to_csv(PASTA_RESULTADOS / "metricas_teste_limiar_ajustado.csv")
    coeficientes.to_csv(PASTA_RESULTADOS / "coeficientes_regressao_logistica.csv")
    importancias.to_csv(PASTA_RESULTADOS / "importancias_random_forest.csv")
    with open(PASTA_RESULTADOS / "relatorios_classificacao.txt", "w", encoding="utf-8") as arquivo:
        for nome, texto in relatorios.items():
            arquivo.write(f"{nome}\n{texto}\n\n")
    with open(PASTA_RESULTADOS / "hiperparametros.json", "w", encoding="utf-8") as arquivo:
        json.dump(
            {
                "regressao_logistica": busca_rl.best_params_,
                "random_forest": busca_rf.best_params_,
                "limiar_random_forest": round(limiar_rf, 3),
            },
            arquivo,
            ensure_ascii=False,
            indent=2,
        )
    joblib.dump(modelos, PASTA_RESULTADOS / "modelos.joblib")
    print(f"Tudo salvo em {PASTA_RESULTADOS}")


if __name__ == "__main__":
    main()
