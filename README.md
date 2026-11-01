# Tech Challenge Fase 1, diagnóstico de diabetes com machine learning

Esse é o meu Tech Challenge da Fase 1 da Pós Tech IA para Devs da FIAP. Montei um modelo de classificação que ajuda na triagem de diabetes a partir de dados clínicos simples, como glicose, IMC e idade. A proposta é apoiar o médico apontando quem precisa de um olhar mais cuidadoso, e não substituir o diagnóstico.

## A base

Usei a Pima Indians Diabetes, a base indicada no enunciado ([Kaggle](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)). Ela está em `data/diabetes.csv`.

- 768 pacientes, todas mulheres com 21 anos ou mais da população Pima
- 8 variáveis (gestações, glicose do teste oral de 2 horas, pressão diastólica, dobra cutânea, insulina, IMC, histórico familiar e idade)
- Alvo `Outcome`, com 500 pacientes sem diabetes (65%) e 268 com diabetes (35%)

## O que eu fiz

1. **Exploração dos dados** com estatísticas, histogramas e boxplots por grupo, correlação de Pearson e gráfico de dispersão
2. **Tratamento dos zeros impossíveis** em glicose, pressão, dobra cutânea, insulina e IMC, que viram dado faltando e são preenchidos pela mediana do treino
3. **Separação estratificada** em 80% treino e 20% teste, com o teste usado uma vez só no final
4. **Dois modelos dentro de Pipeline**, Regressão Logística (com StandardScaler) e Random Forest, comparados com um modelo base que chuta sempre a classe mais comum
5. **Ajuste de hiperparâmetros** com GridSearchCV e validação cruzada estratificada de 5 dobras
6. **Avaliação** com acurácia, precisão, recall, F1 e ROC AUC, matriz de confusão, classification report e curva ROC
7. **Ajuste do limiar de decisão** escolhido na validação cruzada para priorizar o recall
8. **Interpretação** com coeficientes, feature importance, importância por permutação e SHAP

## Resultados no teste (154 pacientes)

| Modelo | Acurácia | Precisão | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Regressão Logística | 0,708 | 0,571 | 0,667 | 0,615 | 0,810 |
| Random Forest | 0,734 | 0,597 | 0,741 | 0,661 | 0,814 |
| Random Forest com limiar 0,46 | 0,721 | 0,573 | 0,796 | 0,667 | 0,814 |

A métrica que mais importa aqui é o recall de quem tem diabetes, porque o pior erro numa triagem é mandar para casa uma paciente doente. A Random Forest com o limiar ajustado encontrou 43 das 54 pacientes com diabetes do teste. Glicose, IMC e idade foram as variáveis que mais pesaram nos dois modelos.

O detalhe completo de cada etapa e a discussão sobre os limites do modelo estão no notebook e no relatório técnico.

## Estrutura do repositório

```
data/diabetes.csv                      base usada
notebooks/tech_challenge_diabetes.ipynb análise completa com gráficos e explicações
src/main.py                            roda tudo do zero e salva os resultados
src/diabetes/config.py                 caminhos, colunas e parâmetros fixos
src/diabetes/dados.py                  carga, tratamento dos zeros e separação treino e teste
src/diabetes/exploracao.py             gráficos da análise exploratória
src/diabetes/modelagem.py              pipelines, grades de hiperparâmetros e validação cruzada
src/diabetes/avaliacao.py              métricas, gráficos de avaliação, limiar e SHAP
resultados/                            métricas, gráficos e modelos gerados pelo main.py
docs/relatorio_tecnico.pdf             relatório técnico
Dockerfile                             ambiente para rodar o projeto em container
requirements.txt                       bibliotecas com versão fixa
```

## Como rodar com Docker

Construir a imagem

```bash
docker build -t tech-challenge-fase1 .
```

Rodar o pipeline completo. Os resultados aparecem na pasta `resultados` da sua máquina.

```bash
docker run --rm -v "$(pwd)/resultados:/app/resultados" tech-challenge-fase1
```

No Windows pelo PowerShell troque `$(pwd)` por `${PWD}`.

Abrir o notebook no navegador

```bash
docker run --rm -p 8888:8888 tech-challenge-fase1 jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser --notebook-dir=/app/notebooks
```

Depois é só abrir o link com o token que aparece no terminal.

## Como rodar sem Docker

Precisa do Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

No Windows ative o ambiente com `.venv\Scripts\activate`.

## Cuidados que tomei

- Toda a preparação (mediana e escala) fica dentro do Pipeline e é ajustada só no treino de cada dobra, então nenhuma informação do teste vaza para o modelo
- A divisão e a validação cruzada são estratificadas e usam `random_state=42`, então quem rodar de novo chega nos mesmos números
- O limiar foi escolhido com as probabilidades da validação cruzada, nunca olhando o teste
- Os dois modelos foram comparados pela média e pelo desvio das dobras, e não por uma única divisão

## Limitações

O modelo aprendeu com mulheres Pima de 21 anos ou mais e com glicose medida no teste de 2 horas. Ele não deve ser usado em outra população ou com glicose de jejum sem antes ser validado de novo. Ele serve para apoiar a triagem e o médico continua com a palavra final.
