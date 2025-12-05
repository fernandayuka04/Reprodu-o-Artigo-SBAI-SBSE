# ==============================================================================
# SCRIPT 5/6: Autoencoder (AE) SEM SMOTE
# Objetivo: Treinar e avaliar o AE usando apenas a classe normal (0)
#           sem aumentar o número de amostras (None Sampling).
# Dependência: tensorflow/keras
# ==============================================================================

# Importações necessárias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score
from scipy.interpolate import interp1d
import os
# Importações do Keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf # Para configurar a seed

# --- Configurações Iniciais ---
SEED = 42
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(SEED) # Configura a seed do TensorFlow
DATA_PATH = os.path.join('../data set', 'data set.csv')

# --- Funções de Preparação de Dados e Avaliação ---

def load_sgcc_data(path):
    """Carrega dados ou gera dados simulados em caso de falha."""
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
    """Realiza imputação linear de NaNs e normaliza os dados com MinMaxScaler."""
    X_imputed = X_data.copy()
    for i in range(X_imputed.shape[0]):
        series = X_imputed[i, :]
        not_nan_indices = np.where(~np.isnan(series))[0]
        nan_indices = np.where(np.isnan(series))[0]
        if len(not_nan_indices) >= 2:
            # Interpolação linear para preencher NaNs
            interp_func = interp1d(not_nan_indices, series[not_nan_indices], kind='linear', fill_value='extrapolate')
            series[nan_indices] = interp_func(nan_indices)
        series[np.isnan(series)] = 0
        X_imputed[i, :] = series

    # Tratamento básico de outliers (clipped no 2*std acima da média)
    avg_x = np.mean(X_imputed, axis=0)
    std_x = np.std(X_imputed, axis=0)
    threshold = avg_x + 2 * std_x
    X_outlier_handled = X_imputed.copy()
    for j in range(X_outlier_handled.shape[1]):
        mask = X_outlier_handled[:, j] > threshold[j]
        X_outlier_handled[mask, j] = threshold[j]

    # Normalização
    if fit_scaler:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_outlier_handled)
        return X_scaled, scaler
    else:
        X_scaled = scaler.transform(X_outlier_handled)
        return X_scaled, scaler

def print_results(title, results):
    """Imprime os resultados detalhados das métricas."""
    print(f"\n--- {title} ---")
    print("Métrica | Classe 0 (Normal) | Classe 1 (Fraude) | Média Macro")
    print("-" * 65)

    # Imprime Acurácia (é um valor único, repete para o formato)
    acc = results['Acc']
    print(f"Acc.    | {acc:.2f}% | {acc:.2f}% | {acc:.2f}%")

    # Imprime Precisão (Precision)
    prec_0, prec_1, prec_avg = results['Prec(0)'], results['Prec(1)'], results['Prec(avg)']
    print(f"Prec.   | {prec_0:.2f}% | {prec_1:.2f}% | {prec_avg:.2f}%")

    # Imprime Recall
    rec_0, rec_1, rec_avg = results['Rec(0)'], results['Rec(1)'], results['Rec(avg)']
    print(f"Recall  | {rec_0:.2f}% | {rec_1:.2f}% | {rec_avg:.2f}%")

    # Imprime F1-Score
    f1_0, f1_1, f1_avg = results['F1(0)'], results['F1(1)'], results['F1(avg)']
    print(f"F1-Score| {f1_0:.2f}% | {f1_1:.2f}% | {f1_avg:.2f}%")
    print("-" * 65)

def create_autoencoder(input_dim):
    """Constrói o modelo Autoencoder (Arquitetura: 32, 16, 8, 16, 32)."""
    # Encoder
    input_layer = Input(shape=(input_dim,)) # Camada de entrada com a dimensão dos dados (N_DAYS)
    encoder = Dense(32, activation='relu', name='encoder_1')(input_layer)
    encoder = Dense(16, activation='relu', name='encoder_2')(encoder)
    latent_space = Dense(8, activation='relu', name='latent_space')(encoder) # Espaço latente (menor dimensão)

    # Decoder
    decoder = Dense(16, activation='relu', name='decoder_1')(latent_space)
    decoder = Dense(32, activation='relu', name='decoder_2')(decoder)
    output_layer = Dense(input_dim, activation='linear', name='output_layer')(decoder) # Saída com a dimensão original

    autoencoder = Model(inputs=input_layer, outputs=output_layer)

    # Compilação: Adam com LR 0.001 (do artigo), perda MSE (para reconstrução)
    optimizer = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=optimizer, loss='mse')

    return autoencoder

def evaluate_autoencoder_model(model, X_test, y_test, threshold):
    """
    Avalia o AE baseado no erro de reconstrução e calcula as métricas detalhadas.
    """
    # 1. Reconstrução
    X_reconstructed = model.predict(X_test, verbose=0)
    # 2. Cálculo do Erro de Reconstrução
    reconstruction_error = np.mean(np.square(X_test - X_reconstructed), axis=1)
    # 3. Predição
    # Predição: Se o erro > threshold (anomalia), rótulo = 1 (fraude)
    y_pred = (reconstruction_error > threshold).astype(int)

    # 4. Cálculo das Métricas (usando sklearn.metrics)

    # Métrica: Acurácia geral
    accuracy = accuracy_score(y_test, y_pred) * 100

    # Métricas: Média Macro
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100

    # Métricas: Por Classe (usando average=None)
    prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0) * 100
    rec_per_class = recall_score(y_test, y_pred, average=None, zero_division=0) * 100
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0) * 100

    results = {
        'Acc': accuracy,
        'Prec(avg)': prec_macro,
        'Rec(avg)': rec_macro,
        'F1(avg)': f1_macro,
        # Resultados por classe: [Classe 0, Classe 1]
        'Prec(0)': prec_per_class[0],
        'Prec(1)': prec_per_class[1],
        'Rec(0)': rec_per_class[0],
        'Rec(1)': rec_per_class[1],
        'F1(0)': f1_per_class[0],
        'F1(1)': f1_per_class[1],
    }
    return results

# --- Execução Principal ---

# 1. Carregar e Dividir Dados
X_raw, y, N_DAYS = load_sgcc_data(DATA_PATH)
X_train_full, X_test, y_train_full, y_test = train_test_split(X_raw, y, test_size=0.3, random_state=SEED, stratify=y)
X_train_normal = X_train_full[y_train_full == 0] # Filtra apenas a classe normal para treino

# 2. Pré-processamento
X_train_normal_scaled, scaler = preprocess_data(X_train_normal, fit_scaler=True)
X_test_scaled, _ = preprocess_data(X_test, fit_scaler=False, scaler=scaler)

# 3. Configuração e Treinamento do Autoencoder (Parâmetros do Artigo)
INPUT_DIM = X_train_normal_scaled.shape[1]
AE_THRESHOLD = 0.05 # Limiar para definir anomalia (erro de reconstrução > 0.05)

model_ae_none = create_autoencoder(INPUT_DIM)

print("\n--- AE (None Sampling) - Configuração ---")
print(f"Arquitetura: (32, 16, 8, 16, 32)")
print("Treinamento: 50 Epochs, 32 Batch Size, Loss: MSE")
print(f"Limiar de Anomalia (AE_THRESHOLD): {AE_THRESHOLD}")
print(f"Treinamento em {X_train_normal_scaled.shape[0]} amostras normais.")

# Treinamento: O input é igual ao target (reconstrução da entrada)
model_ae_none.fit(X_train_normal_scaled, X_train_normal_scaled,
                  epochs=50,
                  batch_size=32,
                  validation_split=0.2,
                  verbose=0)

# 4. Avaliação
results_ae_none = evaluate_autoencoder_model(model_ae_none, X_test_scaled, y_test, AE_THRESHOLD)
print_results("AE (None Sampling) - Resultados Detalhados", results_ae_none)
