# 1. Importar as bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
# A classe específica para classificação no LightGBM
import lightgbm as lgb
from sklearn.metrics import accuracy_score

# --- Preparação dos Dados (Exemplo com o dataset Iris) ---

# 2. Carregar ou criar um dataset (reutilizando o Iris)
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data   # Variáveis independentes (features)
y = iris.target # Variável dependente (rótulos/classes)

# 3. Dividir os dados em conjuntos de Treinamento e Teste
X_train, X_test, y_train, y_test = train_test_split(
    X,                      # O conjunto completo de features
    y,                      # Os rótulos/targets correspondentes
    test_size=0.3,          # 30% dos dados serão usados para teste
    random_state=42         # Garante a reprodutibilidade da divisão
)

# --- Construção e Treinamento do Modelo LightGBM ---

# 4. Inicializar o modelo LGBM Classifier
# n_estimators: Número de iterações de boosting (o número de árvores)
# num_leaves: O número máximo de folhas em cada árvore (ajuda a controlar a complexidade)
# random_state: Para reprodutibilidade
lgbm_clf = lgb.LGBMClassifier(
    n_estimators=100,
    num_leaves=31,
    random_state=42,
    verbose=-1 # Silencia a saída de logs do LightGBM
)

# 5. Treinar o modelo
# O modelo constrói árvores sequencialmente, corrigindo os erros das árvores anteriores (boosting)
# Ele usa a técnica de GOSS e EFB para otimizar o processo (explicado na seção final)
lgbm_clf.fit(
    X_train,
    y_train,
    # Parâmetros adicionais para monitoramento durante o treinamento (opcional)
    eval_set=[(X_test, y_test)], # Conjunto para avaliação (monitora o desempenho)
    eval_metric='multi_logloss', # Métrica de avaliação para classificação multi-classe
    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)] # Parada antecipada para evitar overfitting
)

# --- Avaliação do Modelo ---

# 6. Fazer previsões no conjunto de teste
y_pred = lgbm_clf.predict(X_test)

# 7. Avaliar o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)

# 8. Imprimir os resultados
print(f"Número de iterações de boosting (n_estimators): {lgbm_clf.get_params()['n_estimators']}")
print(f"Acurácia do modelo LightGBM: {accuracy:.4f}")

# --- Análise Adicional (Importância dos Atributos) ---

# 9. Obter a importância de cada feature
# LightGBM pode retornar a importância dos atributos com base no "split" (vezes que a feature foi usada)
feature_importances = pd.Series(
    lgbm_clf.feature_importances_,
    index=iris.feature_names
).sort_values(ascending=False)

# 10. Imprimir a importância dos atributos
print("\nImportância dos Atributos:")
print(feature_importances)
