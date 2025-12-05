import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import warnings


# Ignorar FutureWarning do Matplotlib
warnings.filterwarnings("ignore", category=FutureWarning)


# ==============================================================================
# 1. Geração de Dados Simulados (Estrutura SGCC)
# O artigo usa a base SGCC, que contém séries temporais de consumo e rótulos (FLAG).
# Aqui, simulamos um DataFrame com 30 dias de consumo e a coluna FLAG.
# ==============================================================================
N_SAMPLES = 1000  # Total de consumidores
ANOMALY_RATIO = 0.05 # 5% de anomalias (muito desbalanceado, como na vida real)


print("1. Gerando dados simulados com estrutura semelhante à SGCC...")


# Gerar dados normais e anomalias
np.random.seed(42)
data = {}
# 30 colunas de consumo diário (C_Dia_1 a C_Dia_30)
for i in range(1, 31):
    # Consumo normal em torno de 100 com ruído
    data[f'C_Dia_{i}'] = np.random.normal(100, 15, N_SAMPLES)


# Gerar a coluna FLAG (Target)
flags = np.zeros(N_SAMPLES, dtype=int)
num_anomalies = int(N_SAMPLES * ANOMALY_RATIO)


# Selecionar aleatoriamente 5% dos consumidores para serem anomalias (roubo)
anomaly_indices = np.random.choice(N_SAMPLES, num_anomalies, replace=False)
flags[anomaly_indices] = 1
data['FLAG'] = flags


df = pd.DataFrame(data)


# Introduzir alguns valores ausentes artificialmente para demonstrar a Interpolação Linear
# --- LINHA CORRIGIDA ABAIXO ---
# Correção: O indexador .iloc foi substituído por .loc porque a seleção de colunas
# np.random.choice(df.columns[:-1], ...) retorna nomes de colunas (strings/labels),
# e não posições (inteiros) exigidas pelo .iloc.
df.loc[np.random.choice(df.index, size=50, replace=False), np.random.choice(df.columns[:-1], size=10, replace=False)] = np.nan
# --- FIM DA LINHA CORRIGIDA ---


print(f"Dimensões do DataFrame simulado: {df.shape}")
print(f"Contagem de classes antes do SMOTE:\n{df['FLAG'].value_counts()}")


# ==============================================================================
# 2. Pré-processamento (Conforme o Artigo)
# ==============================================================================
X = df.drop('FLAG', axis=1)
Y = df['FLAG']


# A. Interpolação Linear para lidar com valores ausentes
print("\n2A. Aplicando Interpolação Linear para valores ausentes...")
# O método 'interpolate(method='linear', axis=0)' preenche os NaNs
# usando valores lineares entre os pontos conhecidos.
X = X.interpolate(method='linear', axis=0)


# 3. Divisão dos Dados
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)


# B. SMOTE (Synthetic Minority Over-sampling Technique) para balanceamento de classes
print("\n2B. Aplicando SMOTE no conjunto de treino para balanceamento...")
smote = SMOTE(random_state=42)
X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)


print(f"Contagem de classes após SMOTE no treino:\n{Y_train_res.value_counts()}")


# ==============================================================================
# 3. Treinamento e Avaliação do Random Forest (RF)
# ==============================================================================
print("\n--- 3. Random Forest (RF) ---")


# Parâmetros de exemplo (Ajustados para um bom desempenho em fraudes)
# Estes são os hiperparâmetros (que você perguntou onde colocar)
RF_PARAMS = {
    'n_estimators': 200,             # Número de árvores na floresta
    'max_depth': 10,                 # Profundidade máxima da árvore
    'min_samples_leaf': 5,           # Número mínimo de amostras necessárias em um nó folha
    'random_state': 42,
    'class_weight': 'balanced'       # Pesa as amostras inversamente proporcionalmente à frequência
}


rf_model = RandomForestClassifier(**RF_PARAMS) # Colocando os parâmetros aqui
rf_model.fit(X_train_res, Y_train_res)
rf_proba = rf_model.predict_proba(X_test)[:, 1]


print(f"AUC-ROC (RF): {roc_auc_score(Y_test, rf_proba):.4f}")


# Exemplo de Relatório (usando o limite padrão de 0.5)
rf_pred = (rf_proba > 0.5).astype(int)
print("\nRelatório de Classificação (RF):")
print(classification_report(Y_test, rf_pred, target_names=['Normal (0)', 'Anomalia (1)']))




# ==============================================================================
# 4. Treinamento e Avaliação do Light GBM (LGBM)
# ==============================================================================
print("\n--- 4. Light Gradient Boost Machine (LGBM) ---")


# Parâmetros de exemplo (Otimizados para velocidade e precisão)
# Estes são os hiperparâmetros (que você perguntou onde colocar)
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 200,             # Número de iterações/árvores
    'learning_rate': 0.05,
    'num_leaves': 31,                # Número de folhas por árvore
    'max_depth': 7,                  # Profundidade máxima da árvore
    'random_state': 42,
    'n_jobs': -1
}


lgb_model = lgb.LGBMClassifier(**LGBM_PARAMS) # Colocando os parâmetros aqui
lgb_model.fit(X_train_res, Y_train_res)
lgb_proba = lgb_model.predict_proba(X_test)[:, 1]


print(f"AUC-ROC (LGBM): {roc_auc_score(Y_test, lgb_proba):.4f}")


# Exemplo de Relatório (usando o limite padrão de 0.5)
lgb_pred = (lgb_proba > 0.5).astype(int)
print("\nRelatório de Classificação (LGBM):")
print(classification_report(Y_test, lgb_pred, target_names=['Normal (0)', 'Anomalia (1)']))