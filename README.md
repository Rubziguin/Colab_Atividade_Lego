# 🧱 Previsão de Preços de Sets LEGO com Redes Neurais

LAB da disciplina de **Inteligência Artificial**, Ciência da Computação, **PUC-SP**.

O objetivo é completar o preço dos sets LEGO que não têm preço na base usando uma **Rede Neural Artificial (ANN)** do Scikit-Learn.

**Dupla:** _Nome 1_ · _Nome 2_

---

## 📊 Dados

[LEGO Sets, Maven Analytics](https://mavenanalytics.io/data-playground/lego-sets). São cerca de 18 mil sets lançados entre 1970 e 2022, com tema, número de peças, minifiguras, idade recomendada e preço de varejo nos EUA (`US_retailPrice`).

Cerca de **60% dos sets não têm preço** registrado.

## 🔧 Etapas

1. **Carregamento e EDA:** dados carregados com Pandas e análise dos valores faltantes com `missingno`.
2. **Limpeza:**
   - remoção de URLs e duplicados;
   - tratamento de faltantes (`minifigs` → 0, rótulos para categorias vazias);
   - correção de valores inválidos (idade mínima > 18);
   - criação de indicadores de faltante.
3. **Análise das variáveis:** correlação e gráficos para identificar o que explica o preço.
4. **Modelagem:** `MLPRegressor` dentro de um `Pipeline` que faz:
   - preenchimento dos faltantes pela mediana, `log1p` e `StandardScaler` nas variáveis numéricas;
   - One-Hot Encoding nas categóricas;
   - previsão do alvo em escala log (`TransformedTargetRegressor`);
   - escolha dos hiperparâmetros com `GridSearchCV`.
5. **Avaliação:** comparação com um modelo que sempre prevê a mediana e com uma regressão linear (Ridge), mais a importância das variáveis por permutação.
6. **Coluna `price`:** usa o preço real quando existe e o previsto pela ANN quando falta. A coluna `price_source` identifica a origem.
7. **Visualizações:** preço real x previsto, evolução por ano, preço por peça e temas mais caros.

## 📈 Resultados (conjunto de teste)

| Modelo | MAE (US$) | R² |
|---|---|---|
| Mediana (referência) | ~26 | ~0 |
| Regressão linear (Ridge) | ~9,9 | ~0,80 |
| **Rede Neural (MLP)** | **~7,4** | **~0,92** |

As variáveis mais importantes foram o **número de peças**, o **tema/grupo de tema**, a **categoria** e as **minifiguras**. Depois do modelo, **100% dos sets** ficaram com preço.

> Os valores podem variar um pouco a cada execução.

## ▶️ Como executar

Abra o notebook no **Google Colab** ou no Jupyter e rode todas as células (**Run All**). Os dados são baixados automaticamente com `wget`.

Dependências:

```bash
pip install pandas numpy matplotlib seaborn missingno scikit-learn
```

## 📁 Arquivos

- `PUCSP_CS_AI_10_AI_ML_ANN_LEGO_LAB.ipynb`: notebook com todo o LAB.
- `lego_sets_com_preco.csv`: gerado ao executar o notebook, com a coluna `price` completa.

## 🛠️ Tecnologias

Python · Pandas · NumPy · Matplotlib · Seaborn · missingno · Scikit-Learn
