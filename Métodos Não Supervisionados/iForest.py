# ==============================================================================
# SCRIPT 3/6: Isolation Forest (iForest) SEM SMOTE
# Objetivo: Treinar e avaliar o iForest usando apenas a classe normal (0)
#           sem aumentar o número de amostras (None Sampling).
# ==============================================================================

# 1. Importações necessárias
import numpy as np
import pandas as pd
from collections import Counter # Necessário para visualizar a distribuição
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest # O modelo de floresta de isolamento
# Módulos para cálculo das métricas
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score
from scipy.interpolate import interp1d
import os
import tensorflow as tf # Para configurar a seed (mantido por consistência de SEED)

# 2. Configurações Iniciais
SEED = 42
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(SEED)
# Simulação do caminho do arquivo de dados
DATA_PATH = os.path.join('.', 'data set.csv') # Ajustado o path para 'data set.csv'

# --- 3. Funções de Suporte (Completas e Detalhadas) ---

def load_sgcc_data(path):
    """Carrega o dataset real ou gera dados simulados se o arquivo não for encontrado."""
    try:
        df = pd.read_csv(path)
        TARGET_COLUMN = df.columns[-1]
        y = df[TARGET_COLUMN].values
        X_data_raw = df.drop(columns=[TARGET_COLUMN])
        X_data = X_data_raw.apply(pd.to_numeric, errors='coerce').values
        N_DAYS = X_data.shape[1]
        print(f"Dados carregados: {X_data.shape[0]} consumidores, {N_DAYS} dias.")
        return X_data, y, N_DAYS
    except FileNotFoundError:
        # Lógica de SIMULAÇÃO
        N_SAMPLES = 1000; N_DAYS_SIM = 1035; FRAUD_RATE = 0.0853
        time = np.linspace(0, 2 * np.pi, N_DAYS_SIM)
        X_normal = np.tile(50 + 20 * np.sin(time * 5), (int(N_SAMPLES * (1 - FRAUD_RATE)), 1)) + np.random.normal(0, 5, (int(N_SAMPLES * (1 - FRAUD_RATE)), N_DAYS_SIM))
        X_fraud = np.tile(50 + 20 * np.sin(time * 5), (int(N_SAMPLES * FRAUD_RATE), 1))
        for i in range(X_fraud.shape[0]):
            X_fraud[i, :np.random.randint(200, N_DAYS_SIM)] *= np.random.uniform(0.1, 0.5)
            X_fraud[i, :] += np.random.normal(0, 2, N_DAYS_SIM)
        X_sim = np.vstack([X_normal, X_fraud]); y_sim = np.array([0] * X_normal.shape[0] + [1] * X_fraud.shape[0])
        print("ERRO: Arquivo não encontrado. Usando dados SIMULADOS.")
        return X_sim, y_sim, N_DAYS_SIM

def preprocess_data(X_data, fit_scaler=False, scaler=None):
    """Realiza interpolação linear, tratamento de outliers e normalização MinMaxScaler."""
    # Tratamento de valores ausentes (NaN) por interpolação linear
    X_imputed = X_data.copy()
    for i in range(X_imputed.shape[0]):
        series = X_imputed[i, :]
        not_nan_indices = np.where(~np.isnan(series))[0]
        nan_indices = np.where(np.isnan(series))[0]
        if len(not_nan_indices) >= 2:
            interp_func = interp1d(not_nan_indices, series[not_nan_indices], kind='linear', fill_value='extrapolate')
            series[nan_indices] = interp_func(nan_indices)
        series[np.isnan(series)] = 0
        X_imputed[i, :] = series

    # Tratamento de Outliers (clipagem em Média + 2*Desvio Padrão)
    avg_x = np.mean(X_imputed, axis=0)
    std_x = np.std(X_imputed, axis=0)
    threshold = avg_x + 2 * std_x
    X_outlier_handled = X_imputed.copy()

    for j in range(X_outlier_handled.shape[1]):
        mask = X_outlier_handled[:, j] > threshold[j]
        X_outlier_handled[mask, j] = threshold[j]

    # Normalização (MinMaxScaler)
    if fit_scaler:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_outlier_handled)
        return X_scaled, scaler
    else:
        X_scaled = scaler.transform(X_outlier_handled)
        return X_scaled, scaler

def evaluate_anomaly_model(model, X_test, y_test):
    """
    Avalia o iForest (ou outro modelo de anomalia sklearn) e calcula as métricas detalhadas.
    O iForest retorna +1 (Normal) e -1 (Anomalia).
    """
    # Predição: +1 (Normal), -1 (Anomalia)
    y_pred_sklearn = model.predict(X_test)

    # 1. Mapeamento dos rótulos: -1 (Anomalia/Fraude) -> 1; +1 (Normal) -> 0
    # O modelo agora está em conformidade com os rótulos 0 e 1 do y_test.
    y_pred = np.where(y_pred_sklearn == -1, 1, 0) # Linha 116 (Mapeamento)

    # 2. Cálculo das Métricas (usando sklearn.metrics)

    # Métrica: Acurácia geral
    accuracy = accuracy_score(y_test, y_pred) * 100 # Linha 121

    # Métricas: Média Macro (Prec(avg), Rec(avg), F1(avg))
    # average='macro' calcula a média não ponderada das classes (média relevante para desbalanceamento)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100 # Linha 125
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100     # Linha 126
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100          # Linha 127

    # Métricas: Por Classe (usando average=None) -> Permite determinar Maior/Menor
    # average=None retorna um array: [Valor Classe 0, Valor Classe 1]
    prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0) * 100 # Linha 130
    rec_per_class = recall_score(y_test, y_pred, average=None, zero_division=0) * 100     # Linha 131
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0) * 100          # Linha 132

    results = {
        'Acc': accuracy,     # Linha 135
        'Prec(avg)': prec_macro, # Linha 136
        'Rec(avg)': rec_macro,   # Linha 137
        'F1(avg)': f1_macro,     # Linha 138
        # Resultados por classe (usados para preencher as colunas Maior e Menor)
        'Prec(0)': prec_per_class[0], # Linha 140
        'Prec(1)': prec_per_class[1], # Linha 141
        'Rec(0)': rec_per_class[0],   # Linha 142
        'Rec(1)': rec_per_class[1],   # Linha 143
        'F1(0)': f1_per_class[0],     # Linha 144
        'F1(1)': f1_per_class[1],     # Linha 145
    }
    return results

def print_results(title, results):
    """Imprime os resultados detalhados das métricas no formato de tabela."""
    print(f"\n--- {title} ---")
    print("Métrica | Classe 0 (Normal) | Classe 1 (Fraude) | Média Macro") # Linha 150 (Cabeçalho)
    print("-" * 65)

    # Imprime Acurácia (é um valor único)
    acc = results['Acc']
    print(f"Acc.    | {acc:.2f}% | {acc:.2f}% | {acc:.2f}%") # Linha 154

    # Imprime Precisão (Precision)
    prec_0, prec_1, prec_avg = results['Prec(0)'], results['Prec(1)'], results['Prec(avg)']
    print(f"Prec.   | {prec_0:.2f}% | {prec_1:.2f}% | {prec_avg:.2f}%") # Linha 158

    # Imprime Recall
    rec_0, rec_1, rec_avg = results['Rec(0)'], results['Rec(1)'], results['Rec(avg)']
    print(f"Recall  | {rec_0:.2f}% | {rec_1:.2f}% | {rec_avg:.2f}%") # Linha 162

    # Imprime F1-Score
    f1_0, f1_1, f1_avg = results['F1(0)'], results['F1(1)'], results['F1(avg)']
    print(f"F1-Score| {f1_0:.2f}% | {f1_1:.2f}% | {f1_avg:.2f}%") # Linha 166
    print("-" * 65)

# --- 4. Execução Principal ---

# 4.1 Carregar e Dividir Dados
X_raw, y, N_DAYS = load_sgcc_data(DATA_PATH) # Linha 172
# Separa 70% para treino e 30% para teste, mantendo a proporção de classes (stratify).
X_train_full, X_test, y_train_full, y_test = train_test_split(X_raw, y, test_size=0.3, random_state=SEED, stratify=y) # Linha 174
# Treinamento apenas com a classe normal (0) - None Sampling
X_train_normal = X_train_full[y_train_full == 0] # Linha 175

# 4.2 Pré-processamento
X_train_normal_scaled, scaler = preprocess_data(X_train_normal, fit_scaler=True) # Linha 178
X_test_scaled, _ = preprocess_data(X_test, fit_scaler=False, scaler=scaler) # Linha 179

# 4.3 Configuração e Treinamento do iForest (Parâmetros do Artigo)
IFOREST_PARAMS = {
    'n_estimators': 100,
    'max_samples': 'auto',
    # Contamination é crucial: Estima a proporção de outliers no conjunto de treino.
    # O Isolation Forest usa esse valor para definir automaticamente o limiar de decisão.
    'contamination': 0.1,
    'random_state': SEED
}
print("\n--- iForest (None Sampling) - Configuração ---") # Linha 193
print("Parâmetros:", IFOREST_PARAMS) # Linha 194
print(f"Treinamento em {X_train_normal_scaled.shape[0]} amostras normais.") # Linha 195

model_iforest_none = IsolationForest(**IFOREST_PARAMS) # Linha 197
model_iforest_none.fit(X_train_normal_scaled) # Treina o modelo, Linha 198

# 4.4 Avaliação
# O iForest já tem seu limiar interno definido pelo parâmetro 'contamination'.
results_iforest_none = evaluate_anomaly_model(model_iforest_none, X_test_scaled, y_test) # Linha 202
print_results("iForest (None Sampling) - Resultados Detalhados", results_iforest_none) # Linha 203
