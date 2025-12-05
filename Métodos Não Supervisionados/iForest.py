# ==============================================================================
# SCRIPT 3/6: Isolation Forest (iForest) SEM SMOTE
# Objetivo: Treinar e avaliar o iForest usando apenas a classe normal (0)
#           sem aumentar o número de amostras (None Sampling).
# ==============================================================================

# Importações necessárias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest # O modelo de floresta de isolamento
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score
from scipy.interpolate import interp1d
import os

# --- Configurações Iniciais ---
SEED = 42
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
DATA_PATH = os.path.join('../data set', 'data set.csv')

# --- Funções de Preparação de Dados e Avaliação (Omitidas por brevidade, mas devem ser incluídas) ---
# (load_sgcc_data, preprocess_data, evaluate_anomaly_model, print_results - Usar as mesmas do Script 1.1)
def load_sgcc_data(path):
    # ... (código da função)
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
    # ... (código da função)
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

    avg_x = np.mean(X_imputed, axis=0)
    std_x = np.std(X_imputed, axis=0)
    threshold = avg_x + 2 * std_x
    X_outlier_handled = X_imputed.copy()

    for j in range(X_outlier_handled.shape[1]):
        mask = X_outlier_handled[:, j] > threshold[j]
        X_outlier_handled[mask, j] = threshold[j]

    if fit_scaler:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_outlier_handled)
        return X_scaled, scaler
    else:
        X_scaled = scaler.transform(X_outlier_handled)
        return X_scaled, scaler
def evaluate_anomaly_model(model, X_test, y_test, threshold, is_ae=False):
    # ... (código da função)
    y_pred_sklearn = model.predict(X_test)
    y_pred = np.where(y_pred_sklearn == -1, 1, 0)
    results = {
        'Acc(avg)': accuracy_score(y_test, y_pred) * 100,
        'Prec(avg)': precision_score(y_test, y_pred, average='macro', zero_division=0) * 100,
        'Rec(avg)': recall_score(y_test, y_pred, average='macro', zero_division=0) * 100,
        'F1(avg)': f1_score(y_test, y_pred, average='macro', zero_division=0) * 100,
        'Rec(1)': recall_score(y_test, y_pred, pos_label=1, zero_division=0) * 100,
    }
    return results
def print_results(title, results):
    # ... (código da função)
    print(f"\n--- {title} ---")
    print(pd.Series(results).apply(lambda x: f'{x:.2f}%'))
    print("-" * 50)

# --- Execução Principal ---

# 1. Carregar e Dividir Dados
X_raw, y, N_DAYS = load_sgcc_data(DATA_PATH)
X_train_full, X_test, y_train_full, y_test = train_test_split(X_raw, y, test_size=0.3, random_state=SEED, stratify=y)
X_train_normal = X_train_full[y_train_full == 0]

# 2. Pré-processamento
X_train_normal_scaled, scaler = preprocess_data(X_train_normal, fit_scaler=True)
X_test_scaled, _ = preprocess_data(X_test, fit_scaler=False, scaler=scaler)

# 3. Configuração e Treinamento do iForest (Parâmetros do Artigo)
IFOREST_PARAMS = {
    'n_estimators': 100,          # Número de árvores na floresta (padrão 100).
    'max_samples': 'auto',        # Número de amostras a serem desenhadas para treinar cada árvore. 'auto' usa min(256, n_samples).
    'contamination': 0.1,         # Estima a proporção de outliers no conjunto de treino (usado para definir o limiar).
    'random_state': SEED          # Seed para reprodutibilidade.
}
IFOREST_THRESHOLD = 0.0 # Limiar de decisão padrão.

print("\n--- iForest (None Sampling) - Configuração ---")
print("Parâmetros:", IFOREST_PARAMS)
print(f"Treinamento em {X_train_normal_scaled.shape[0]} amostras normais.")

model_iforest_none = IsolationForest(**IFOREST_PARAMS)
model_iforest_none.fit(X_train_normal_scaled) # Treina o modelo

# 4. Avaliação
results_iforest_none = evaluate_anomaly_model(model_iforest_none, X_test_scaled, y_test, IFOREST_THRESHOLD)
print_results("iForest (None Sampling) - Resultados", results_iforest_none)