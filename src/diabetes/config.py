from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
CAMINHO_DADOS = RAIZ / "data" / "diabetes.csv"
PASTA_RESULTADOS = RAIZ / "resultados"
PASTA_FIGURAS = PASTA_RESULTADOS / "figuras"

ALVO = "Outcome"

COLUNAS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

COLUNAS_ZERO_INVALIDO = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

NOMES_PT = {
    "Pregnancies": "Gestações",
    "Glucose": "Glicose",
    "BloodPressure": "Pressão diastólica",
    "SkinThickness": "Dobra cutânea",
    "Insulin": "Insulina",
    "BMI": "IMC",
    "DiabetesPedigreeFunction": "Histórico familiar",
    "Age": "Idade",
    "Outcome": "Diabetes",
}

ROTULOS_CLASSE = ["Sem diabetes", "Com diabetes"]

SEMENTE = 42
TAMANHO_TESTE = 0.2
DOBRAS = 5
