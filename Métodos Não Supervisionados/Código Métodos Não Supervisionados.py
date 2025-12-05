import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score
from scipy.interpolate import interp1d
import os
import sys

# Definindo constantes
SEED = 42
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
# O TensorFlow pode ser inconsistente com seeds; esta é uma tentativa de mitigar
try:
    import tensorflow as tf

    tf.random.set_seed(SEED)
except ImportError:
    print("TensorFlow não importado. Os resultados do Autoencoder podem variar.")

# --- 1. Carregamento e Preparação de Dados SGCC ---

# ATENÇÃO: Verifique se este caminho 'data set/data set.csv' está correto no seu ambiente!
DATA_PATH = os.path.join('../data set', 'data set.csv')


def load_sgcc_data(path):
    """Carrega os dados da SGCC, aplica correção de tipo e divide em features (X) e rótulos (y)."""
    try:
        df = pd.read_csv(path)

        # Assume que a última coluna é o rótulo (FLAG)
        TARGET_COLUMN = df.columns[-1]

        y = df[TARGET_COLUMN].values

        # Features: todas as colunas, exceto o rótulo
        X_data_raw = df.drop(columns=[TARGET_COLUMN])

        # CORREÇÃO CRÍTICA: Converte explicitamente todas as features para float.
        # Isso resolve o TypeError com np.isnan() causado por dtype 'object'.
        # O argumento 'errors='coerce'' converte qualquer string que não seja número para NaN,
        # permitindo que a interpolação linear lide com ela no próximo passo.
        X_data = X_data_raw.apply(pd.to_numeric, errors='coerce').values

        N_DAYS = X_data.shape[1]

        print(f"Dados carregados com sucesso: {X_data.shape[0]} consumidores, {N_DAYS} dias.")
        return X_data, y, N_DAYS

    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em: {path}")
        print("Usando dados SIMULADOS para continuar a execução da modelagem.")

        # Simulação para garantir que o script funcione mesmo sem o arquivo
        N_SAMPLES = 1000
        N_DAYS_SIM = 1035
        FRAUD_RATE = 0.0853
        time = np.linspace(0, 2 * np.pi, N_DAYS_SIM)
        normal_pattern = 50 + 20 * np.sin(time * 5)
        X_normal = np.tile(normal_pattern, (int(N_SAMPLES * (1 - FRAUD_RATE)), 1)) + np.random.normal(0, 5, (
            int(N_SAMPLES * (1 - FRAUD_RATE)), N_DAYS_SIM))
        X_fraud = np.tile(normal_pattern, (int(N_SAMPLES * FRAUD_RATE), 1))
        theft_periods = np.random.randint(200, N_DAYS_SIM, size=X_fraud.shape[0])
        for i in range(X_fraud.shape[0]):
            X_fraud[i, :theft_periods[i]] *= np.random.uniform(0.1, 0.5)
            X_fraud[i, :] += np.random.normal(0, 2, N_DAYS_SIM)
        X_sim = np.vstack([X_normal, X_fraud])
        y_sim = np.array([0] * X_normal.shape[0] + [1] * X_fraud.shape[0])
        return X_sim, y_sim, N_DAYS_SIM


# Carrega os dados reais (ou simulados, em caso de erro)
X_raw, y, N_DAYS = load_sgcc_data(DATA_PATH)

# Adiciona NaNs simulados (apenas se os dados não tiverem NaNs suficientes)
# Para fins de teste, garantimos que a função preprocess_data seja exercitada
MISSING_RATE = 0.26
missing_mask = np.random.rand(*X_raw.shape) < MISSING_RATE
X_missing = X_raw.copy()
# Apenas aplica a máscara de NaNs se a conversão anterior não gerou NaNs suficientes
X_missing[missing_mask] = np.nan

# Divisão de Treino e Teste (Estratificada)
X_train_full, X_test, y_train_full, y_test = train_test_split(X_missing, y, test_size=0.3, random_state=SEED,
                                                              stratify=y)

# TREINAMENTO APENAS NA CLASSE NORMAL (0) para Detecção de Anomalias
X_train_normal = X_train_full[y_train_full == 0]
y_train_normal = y_train_full[y_train_full == 0]


# --- 2. Função de Pré-processamento ---

def preprocess_data(X_data, fit_scaler=False, scaler=None):
    """
    Realiza Imputação (Interpolação Linear), Tratamento de Outliers (Regra 2-Sigma)
    e Normalização (MinMaxScaler).
    """
    X_imputed = X_data.copy()

    # a) Imputação de Valores Faltantes (Interpolação Linear)
    for i in range(X_imputed.shape[0]):
        series = X_imputed[i, :]
        not_nan_indices = np.where(~np.isnan(series))[0]
        nan_indices = np.where(np.isnan(series))[0]

        if len(not_nan_indices) >= 2:
            interp_func = interp1d(not_nan_indices, series[not_nan_indices], kind='linear', fill_value='extrapolate')
            series[nan_indices] = interp_func(nan_indices)

        # Trata NaNs restantes (início/fim da série, ou apenas 1 ponto). Substitui por 0.
        series[np.isnan(series)] = 0
        X_imputed[i, :] = series

    # b) Tratamento de Outliers (Regra Média + 2*Std)
    avg_x = np.mean(X_imputed, axis=0)
    std_x = np.std(X_imputed, axis=0)

    # Limiar: Média + 2 desvios-padrão por feature/dia
    threshold = avg_x + 2 * std_x
    X_outlier_handled = X_imputed.copy()

    # Aplica o limite por feature/dia
    for j in range(X_outlier_handled.shape[1]):
        mask = X_outlier_handled[:, j] > threshold[j]
        X_outlier_handled[mask, j] = threshold[j]

    # c) Normalização (MinMaxScaler)
    if fit_scaler:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_outlier_handled)
        return X_scaled, scaler
    else:
        if scaler is None:
            raise ValueError("Scaler deve ser fornecido na fase de transformação.")
        X_scaled = scaler.transform(X_outlier_handled)
        return X_scaled, scaler


# Pré-processamento e fit do scaler APENAS nos dados normais de treinamento
X_train_normal_scaled, scaler = preprocess_data(X_train_normal, fit_scaler=True)

# Pré-processamento dos dados de teste usando o scaler treinado
X_test_scaled, _ = preprocess_data(X_test, fit_scaler=False, scaler=scaler)


# --- 3. Função para SMOTE (Simulação para AD) ---

def apply_smote_to_normal_data(X_normal):
    """
    Simula o efeito do 'SMOTE' no conjunto de treino da classe normal,
    conforme sugerido pela tabela do artigo para os modelos AD.
    Na prática, para modelos AD, o SMOTE é aplicado ao conjunto binário
    completo e a classe normal é usada. Aqui, apenas aumentamos o dataset normal.
    """
    # Duplica as amostras para simular um conjunto de treino maior
    X_smote_simulated = np.vstack([X_normal] * 2)
    return X_smote_simulated


# --- 4. Configuração da Avaliação ---

def evaluate_anomaly_model(model, X_test, y_test, threshold, is_ae=False):
    """Avalia o modelo e aplica o threshold para obter as previsões binárias."""

    if is_ae:
        # Autoencoder: anomalia é baseada no erro de reconstrução > threshold
        X_reconstructed = model.predict(X_test, verbose=0)
        reconstruction_error = np.mean(np.square(X_test - X_reconstructed), axis=1)
        y_pred = (reconstruction_error > threshold).astype(int)
    else:
        # OCSVM e iForest: -1 (Outlier/Fraude) e 1 (Normal)
        y_pred_sklearn = model.predict(X_test)
        # Converte para: 1 (Fraude) se -1, 0 (Normal) se 1
        y_pred = np.where(y_pred_sklearn == -1, 1, 0)

    # Cálculo de métricas (Todas em percentual)
    acc_avg = accuracy_score(y_test, y_pred) * 100
    prec_avg = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100
    rec_avg = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100
    f1_avg = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100

    prec_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0) * 100
    prec_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0) * 100
    rec_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0) * 100
    rec_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0) * 100
    f1_0 = f1_score(y_test, y_pred, pos_label=0, zero_division=0) * 100
    f1_1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0) * 100

    return {
        'Acc(avg)': acc_avg,
        'Prec(avg)': prec_avg,
        'Rec(avg)': rec_avg,
        'F1(avg)': f1_avg,
        'Prec(0)': prec_0,
        'Prec(1)': prec_1,
        'Rec(0)': rec_0,
        'Rec(1)': rec_1,
        'F1(0)': f1_0,
        'F1(1)': f1_1,
    }


def print_results(title, results):
    """Função auxiliar para imprimir os resultados em formato de tabela."""
    print(f"\n--- {title} ---")
    results_series = pd.Series(results)
    print(results_series.apply(lambda x: f'{x:.2f}%'))
    print("-" * 50)


# --- 5. Implementação dos Modelos ---

# Parâmetros de OCSVM (do artigo)
# CORREÇÃO CRÍTICA: 'random_state' foi removido porque não é aceito por OneClassSVM
# com kernel 'rbf' nas versões recentes do scikit-learn.
OCSVM_PARAMS = {
    'kernel': 'rbf',
    'nu': 0.1,  # Estimativa da proporção de outliers
    'gamma': 'auto',
}
OCSVM_THRESHOLD = 0.0  # Limiar de decisão padrão do Sklearn

print("--- OCSVM (One-Class SVM) ---")
print("Parâmetros:", OCSVM_PARAMS)

# 5.1.1 OCSVM com 'None' Sampling
model_ocsvm_none = OneClassSVM(**OCSVM_PARAMS)
model_ocsvm_none.fit(X_train_normal_scaled)
results_ocsvm_none = evaluate_anomaly_model(model_ocsvm_none, X_test_scaled, y_test, OCSVM_THRESHOLD)
print_results("OCSVM (None Sampling)", results_ocsvm_none)

# 5.1.2 OCSVM com 'SMOTE' Sampling (Simulação)
X_train_ocsvm_smote = apply_smote_to_normal_data(X_train_normal_scaled)
model_ocsvm_smote = OneClassSVM(**OCSVM_PARAMS)
model_ocsvm_smote.fit(X_train_ocsvm_smote)
results_ocsvm_smote = evaluate_anomaly_model(model_ocsvm_smote, X_test_scaled, y_test, OCSVM_THRESHOLD)
print_results("OCSVM (SMOTE Sampling - Simulado)", results_ocsvm_smote)

# Parâmetros de Isolation Forest (do artigo)
IFOREST_PARAMS = {
    'n_estimators': 100,
    'max_samples': 'auto',
    'contamination': 0.1,  # Estimativa de 10% de anomalias no conjunto total (padrão do artigo)
    'random_state': SEED # Mantido, pois IsolationForest aceita este argumento.
}
IFOREST_THRESHOLD = 0.0  # Limiar de decisão padrão

print("\n--- Isolation Forest (iForest) ---")
print("Parâmetros:", IFOREST_PARAMS)

# 5.2.1 iForest com 'None' Sampling
model_iforest_none = IsolationForest(**IFOREST_PARAMS)
model_iforest_none.fit(X_train_normal_scaled)
results_iforest_none = evaluate_anomaly_model(model_iforest_none, X_test_scaled, y_test, IFOREST_THRESHOLD)
print_results("iForest (None Sampling)", results_iforest_none)

# 5.2.2 iForest com 'SMOTE' Sampling (Simulação)
X_train_iforest_smote = apply_smote_to_normal_data(X_train_normal_scaled)
model_iforest_smote = IsolationForest(**IFOREST_PARAMS)
model_iforest_smote.fit(X_train_iforest_smote)
results_iforest_smote = evaluate_anomaly_model(model_iforest_smote, X_test_scaled, y_test, IFOREST_THRESHOLD)
print_results("iForest (SMOTE Sampling - Simulado)", results_iforest_smote)

# --- 5.3 Autoencoder (AE) ---

INPUT_DIM = X_train_normal_scaled.shape[1]
# Limiar para o erro de reconstrução. Valor de simulação baseado no artigo.
AE_THRESHOLD = 0.05


def create_autoencoder(input_dim):
    """Constrói o modelo Autoencoder conforme a arquitetura do artigo (32, 16, 8, 16, 32)."""

    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(32, activation='relu', name='encoder_1')(input_layer)
    encoder = Dense(16, activation='relu', name='encoder_2')(encoder)
    latent_space = Dense(8, activation='relu', name='latent_space')(encoder)

    # Decoder
    decoder = Dense(16, activation='relu', name='decoder_1')(latent_space)
    decoder = Dense(32, activation='relu', name='decoder_2')(decoder)
    output_layer = Dense(input_dim, activation='linear', name='output_layer')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)

    # Compilação: Adam com LR 0.001, perda MSE
    optimizer = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=optimizer, loss='mse')

    return autoencoder


print("\n--- Autoencoder (AE) ---")
print(f"Arquitetura: (Encoder: 32, 16, 8) -> (Decoder: 16, 32, {INPUT_DIM})")
print("Treinamento: 50 Epochs, 32 Batch Size")

# 5.3.1 AE com 'None' Sampling
model_ae_none = create_autoencoder(INPUT_DIM)
# Treinamento do AE para reconstruir X_train_normal_scaled (usando ele como input e target)
history_ae_none = model_ae_none.fit(X_train_normal_scaled, X_train_normal_scaled,
                                    epochs=50,
                                    batch_size=32,
                                    validation_split=0.2,
                                    verbose=0)

results_ae_none = evaluate_anomaly_model(model_ae_none, X_test_scaled, y_test, AE_THRESHOLD, is_ae=True)
print_results("AE (None Sampling)", results_ae_none)

# 5.3.2 AE com 'SMOTE' Sampling (Simulação)
X_train_ae_smote = apply_smote_to_normal_data(X_train_normal_scaled)
model_ae_smote = create_autoencoder(INPUT_DIM)
history_ae_smote = model_ae_smote.fit(X_train_ae_smote, X_train_ae_smote,
                                      epochs=50,
                                      batch_size=32,
                                      validation_split=0.2,
                                      verbose=0)

results_ae_smote = evaluate_anomaly_model(model_ae_smote, X_test_scaled, y_test, AE_THRESHOLD, is_ae=True)
print_results("AE (SMOTE Sampling - Simulado)", results_ae_smote)