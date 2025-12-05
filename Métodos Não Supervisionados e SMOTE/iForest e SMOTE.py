# ==============================================================================
# SCRIPT 4/6: Isolation Forest (iForest) COM SMOTE
# Objetivo: Treinar e avaliar o iForest usando o SMOTE para balancear as classes
#           de treino antes de aplicar o modelo.
# Dependência: imblearn, scikit-learn
# ==============================================================================

# Importações necessárias
import pandas as pd  # Manipulação de DataFrames.
import numpy as np  # Operações numéricas.
from collections import Counter  # Contagem de classes.
from imblearn.over_sampling import SMOTE  # Técnica de Sobreamostragem de Minoria Sintética.
from sklearn.preprocessing import MinMaxScaler  # Normalização.
from sklearn.model_selection import train_test_split  # Para separar dados de treino e teste.
from sklearn.ensemble import IsolationForest  # O modelo de floresta de isolamento (iForest).
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score  # Métricas.
from scipy.interpolate import interp1d  # Para preenchimento de valores ausentes (interpolação).
import os

# --- Configurações Iniciais ---
SEED = 42
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
# Simulação do caminho do arquivo de dados (caso você esteja usando um arquivo real)
DATA_PATH = os.path.join('.', 'data set.csv')
IFOREST_THRESHOLD = 0.0  # Limiar de decisão padrão.


# ==============================================================================
# 3. Funções de Suporte (Carregamento, Pré-processamento, Modelo e Avaliação)
# ==============================================================================

def load_sgcc_data(path):
    """Carrega o dataset real ou gera dados simulados se o arquivo não for encontrado."""
    try:
        df = pd.read_csv(path)
        TARGET_COLUMN = df.columns[-1]
        y = df[TARGET_COLUMN].values
        X_data_raw = df.drop(columns=[TARGET_COLUMN])
        # Converte para numpy array
        X_data = X_data_raw.apply(pd.to_numeric, errors='coerce').values
        N_DAYS = X_data.shape[1]
        print(f"Dados carregados: {X_data.shape[0]} consumidores, {N_DAYS} dias.")
        return X_data, y, N_DAYS
    except FileNotFoundError:
        # Lógica de SIMULAÇÃO caso o arquivo não seja encontrado (idêntica à sua referência)
        N_SAMPLES = 1000;
        N_DAYS_SIM = 1035;
        FRAUD_RATE = 0.0853
        time = np.linspace(0, 2 * np.pi, N_DAYS_SIM)
        X_normal = np.tile(50 + 20 * np.sin(time * 5), (int(N_SAMPLES * (1 - FRAUD_RATE)), 1)) + np.random.normal(0, 5,
                                                                                                                  (int(
                                                                                                                      N_SAMPLES * (
                                                                                                                                  1 - FRAUD_RATE)),
                                                                                                                   N_DAYS_SIM))
        X_fraud = np.tile(50 + 20 * np.sin(time * 5), (int(N_SAMPLES * FRAUD_RATE), 1))
        for i in range(X_fraud.shape[0]):
            X_fraud[i, :np.random.randint(200, N_DAYS_SIM)] *= np.random.uniform(0.1, 0.5)
            X_fraud[i, :] += np.random.normal(0, 2, N_DAYS_SIM)
        X_sim = np.vstack([X_normal, X_fraud]);
        y_sim = np.array([0] * X_normal.shape[0] + [1] * X_fraud.shape[0])
        print("ARQUIVO NÃO ENCONTRADO. Usando dados SIMULADOS.")
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
        # Preenche valores NaN restantes (se houver) com zero
        series[np.isnan(series)] = 0
        X_imputed[i, :] = series

    # Tratamento de Outliers (clipagem em Média + 2*Desvio Padrão)
    avg_x = np.mean(X_imputed, axis=0)
    std_x = np.std(X_imputed, axis=0)
    threshold_outlier = avg_x + 2 * std_x
    X_outlier_handled = X_imputed.copy()

    for j in range(X_outlier_handled.shape[1]):
        mask = X_outlier_handled[:, j] > threshold_outlier[j]
        X_outlier_handled[mask, j] = threshold_outlier[j]

    # Normalização (MinMaxScaler)
    if fit_scaler:
        scaler = MinMaxScaler()  # Instancia o MinMaxScaler
        X_scaled = scaler.fit_transform(X_outlier_handled)  # Ajusta e transforma nos dados de treino
        return X_scaled, scaler
    else:
        # Transforma nos dados de teste usando o scaler já ajustado
        X_scaled = scaler.transform(X_outlier_handled)
        return X_scaled, scaler


def evaluate_anomaly_model(model, X_test, y_test, threshold):
    """Avalia o iForest (ou OCSVM) baseado na predição."""
    # iForest retorna 1 para normal e -1 para anomalia/fraude.
    # Convertemos para 0 (normal) e 1 (fraude) para calcular as métricas.
    y_pred_sklearn = model.predict(X_test)
    y_pred = np.where(y_pred_sklearn == -1, 1, 0)

    # Cálculo das métricas
    results = {
        'Acc(avg)': accuracy_score(y_test, y_pred) * 100,
        'Prec(avg)': precision_score(y_test, y_pred, average='macro', zero_division=0) * 100,
        'Rec(avg)': recall_score(y_test, y_pred, average='macro', zero_division=0) * 100,
        'F1(avg)': f1_score(y_test, y_pred, average='macro', zero_division=0) * 100,
        'Rec(1)': recall_score(y_test, y_pred, pos_label=1, zero_division=0) * 100,
    }
    return results


def print_results(title, results):
    """Imprime os resultados formatados."""
    print(f"\n--- {title} ---")
    print(pd.Series(results).apply(lambda x: f'{x:.2f}%'))
    print("-" * 50)


# ==============================================================================
# 4. Execução Principal: Isolation Forest TREINADO com SMOTE
# ==============================================================================

# 4.1 Carregar e Dividir Dados
X_raw, y, N_DAYS = load_sgcc_data(DATA_PATH)  # Carrega os dados (reais ou simulados).
# Separa 70% para treino e 30% para teste, mantendo a proporção de classes (stratify).
X_train_full, X_test, y_train_full, y_test = train_test_split(X_raw, y, test_size=0.3, random_state=SEED, stratify=y)

print("\n" + "=" * 60)
print(f"Dados de Treino (Original): {X_train_full.shape}")
print(f"Distribuição de Classes no Treino: {Counter(y_train_full)}")
print("=" * 60)

# 4.2 Pré-processamento e Normalização (Ajusta o scaler SOMENTE no treino)
X_train_scaled, scaler = preprocess_data(X_train_full, fit_scaler=True)
X_test_scaled, _ = preprocess_data(X_test, fit_scaler=False, scaler=scaler)

# 4.3 Aplicação do SMOTE (Apenas nos dados de TREINO)
print("Aplicando SMOTE aos dados de Treinamento...")
sm = SMOTE(random_state=SEED)  # Instancia o SMOTE.
# Aplica o SMOTE no conjunto de dados de treino normalizado.
X_res, y_res = sm.fit_resample(X_train_scaled, y_train_full)  # Geração de amostras sintéticas.

print("-" * 60)
print("Distribuição de Classes no Treino DEPOIS do SMOTE:", Counter(y_res))
print(f"Número total de amostras no treino depois do SMOTE: {len(X_res)}")
print("-" * 60)

# 4.4 Configuração e Treinamento do Isolation Forest
# Parâmetros de referência:
IFOREST_PARAMS = {
    'n_estimators': 100,
    'max_samples': 'auto',
    # A contaminação deve ser ajustada para o Isolation Forest treinado em dados balanceados.
    # Se o Isolation Forest é usado como detector de anomalias, geralmente é treinado apenas
    # com dados normais (contamination = proporção de anomalias esperadas).
    # Como estamos treinando em dados BALANCEADOS (50/50), o contamination deve ser 0.5.
    'contamination': 0.5,
    'random_state': SEED
}

print("\n--- iForest (SMOTE) - Configuração ---")
print("Parâmetros:", IFOREST_PARAMS)
print(f"Treinamento em {X_res.shape[0]} amostras BALANCEADAS.")

# Treinamento: Utiliza o dataset BALANCEADO (X_res, y_res)
model_iforest_smote = IsolationForest(**IFOREST_PARAMS)
model_iforest_smote.fit(X_res, y_res)

print("Treinamento do iForest (SMOTE) concluído.")

# 4.5 Avaliação
# O parâmetro threshold (limiar) é usado na função evaluate_anomaly_model,
# mas é ignorado pelo iForest que usa seu próprio limiar interno definido
# pelo parâmetro 'contamination'.
results_iforest_smote = evaluate_anomaly_model(model_iforest_smote, X_test_scaled, y_test, IFOREST_THRESHOLD)
print_results("Isolation Forest TREINADO com SMOTE - Resultados no Teste", results_iforest_smote)
print("=" * 60)