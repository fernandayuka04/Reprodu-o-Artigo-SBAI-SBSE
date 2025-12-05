# ==============================================================================
# 1. Importação das Bibliotecas Necessárias
# ==============================================================================
import pandas as pd # Manipulação de DataFrames.
import numpy as np  # Operações numéricas.
from collections import Counter # Contagem de classes.
from imblearn.over_sampling import SMOTE # Técnica de Sobreamostragem de Minoria Sintética.
from sklearn.preprocessing import MinMaxScaler # Normalização: usado na referência do Autoencoder.
from sklearn.model_selection import train_test_split # Para separar dados de treino e teste.
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score # Métricas de avaliação.
from scipy.interpolate import interp1d # Para preenchimento de valores ausentes (interpolação).
import os # Para gerenciamento de caminho de arquivos.

# Importações do Keras (TensorFlow)
from tensorflow.keras.models import Model # Para construir o Autoencoder.
from tensorflow.keras.layers import Input, Dense # Camadas da rede neural.
from tensorflow.keras.optimizers import Adam # Otimizador Adam.
import tensorflow as tf # Para configurar a seed.

# O pipeline será: Carregar/Simular -> Normalização -> SMOTE (no Treino) -> Autoencoder.

# ==============================================================================
# 2. Configurações Iniciais e Variáveis
# ==============================================================================
SEED = 42 # Semente para reprodutibilidade.
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(SEED)
# Simulação do caminho do arquivo de dados (caso você esteja usando um arquivo real)
DATA_PATH = os.path.join('.', 'data set.csv')
AE_THRESHOLD = 0.05 # Limiar de erro de reconstrução definido na sua referência.

# ==============================================================================
# 3. Funções de Suporte (Carregamento, Pré-processamento, Modelo e Avaliação)
# ==============================================================================

def load_sgcc_data(path):
    """Carrega o dataset real ou gera dados simulados se o arquivo não for encontrado."""
    try:
        # Se você tivesse o 'data set.csv' no caminho, ele seria carregado aqui.
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
        N_SAMPLES = 1000; N_DAYS_SIM = 1035; FRAUD_RATE = 0.0853
        time = np.linspace(0, 2 * np.pi, N_DAYS_SIM)
        X_normal = np.tile(50 + 20 * np.sin(time * 5), (int(N_SAMPLES * (1 - FRAUD_RATE)), 1)) + np.random.normal(0, 5, (int(N_SAMPLES * (1 - FRAUD_RATE)), N_DAYS_SIM))
        X_fraud = np.tile(50 + 20 * np.sin(time * 5), (int(N_SAMPLES * FRAUD_RATE), 1))
        for i in range(X_fraud.shape[0]):
            X_fraud[i, :np.random.randint(200, N_DAYS_SIM)] *= np.random.uniform(0.1, 0.5)
            X_fraud[i, :] += np.random.normal(0, 2, N_DAYS_SIM)
        X_sim = np.vstack([X_normal, X_fraud]); y_sim = np.array([0] * X_normal.shape[0] + [1] * X_fraud.shape[0])
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
        scaler = MinMaxScaler() # Instancia o MinMaxScaler
        X_scaled = scaler.fit_transform(X_outlier_handled) # Ajusta e transforma nos dados de treino
        return X_scaled, scaler
    else:
        # Transforma nos dados de teste usando o scaler já ajustado
        X_scaled = scaler.transform(X_outlier_handled)
        return X_scaled, scaler

def print_results(title, results):
    """Imprime os resultados formatados."""
    print(f"\n--- {title} ---")
    print(pd.Series(results).apply(lambda x: f'{x:.2f}%'))
    print("-" * 50)

def create_autoencoder(input_dim):
    """Constrói o modelo Autoencoder (Arquitetura: 32, 16, 8, 16, 32)."""
    # Encoder
    input_layer = Input(shape=(input_dim,)) # Camada de entrada (Input_Dim = N_DAYS)
    encoder = Dense(32, activation='relu', name='encoder_1')(input_layer)
    encoder = Dense(16, activation='relu', name='encoder_2')(encoder)
    latent_space = Dense(8, activation='relu', name='latent_space')(encoder) # Espaço latente

    # Decoder
    decoder = Dense(16, activation='relu', name='decoder_1')(latent_space)
    decoder = Dense(32, activation='relu', name='decoder_2')(decoder)
    output_layer = Dense(input_dim, activation='linear', name='output_layer')(decoder) # Saída

    autoencoder = Model(inputs=input_layer, outputs=output_layer)

    # Compilação: Adam com LR 0.001 (do artigo), perda MSE
    optimizer = Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=optimizer, loss='mse')

    return autoencoder

def evaluate_autoencoder_model(model, X_test, y_test, threshold):
    """Avalia o AE baseado no erro de reconstrução."""
    X_reconstructed = model.predict(X_test, verbose=0)
    # Erro de reconstrução (MSE) por amostra
    reconstruction_error = np.mean(np.square(X_test - X_reconstructed), axis=1)
    # Predição: Se o erro > threshold (anomalia), rótulo = 1 (fraude)
    y_pred = (reconstruction_error > threshold).astype(int)

    # Cálculo das métricas
    results = {
        'Acc(avg)': accuracy_score(y_test, y_pred) * 100,
        'Prec(avg)': precision_score(y_test, y_pred, average='macro', zero_division=0) * 100,
        'Rec(avg)': recall_score(y_test, y_pred, average='macro', zero_division=0) * 100,
        'F1(avg)': f1_score(y_test, y_pred, average='macro', zero_division=0) * 100,
        'Rec(1)': recall_score(y_test, y_pred, pos_label=1, zero_division=0) * 100,
    }
    return results

# ==============================================================================
# 4. Execução Principal: Autoencoder TREINADO com SMOTE
# ==============================================================================

# 4.1 Carregar e Dividir Dados
X_raw, y, N_DAYS = load_sgcc_data(DATA_PATH) # Carrega os dados (reais ou simulados).
# Separa 70% para treino e 30% para teste, mantendo a proporção de classes (stratify).
X_train_full, X_test, y_train_full, y_test = train_test_split(X_raw, y, test_size=0.3, random_state=SEED, stratify=y)

print("\n" + "=" * 60)
print(f"Dados de Treino (Original): {X_train_full.shape}")
print(f"Distribuição de Classes no Treino: {Counter(y_train_full)}")
print(f"Dados de Teste (Original): {X_test.shape}")
print(f"Distribuição de Classes no Teste: {Counter(y_test)}")
print("=" * 60)

# 4.2 Pré-processamento e Normalização (Ajusta o scaler SOMENTE no treino)
X_train_scaled, scaler = preprocess_data(X_train_full, fit_scaler=True)
X_test_scaled, _ = preprocess_data(X_test, fit_scaler=False, scaler=scaler)

# 4.3 Aplicação do SMOTE (Apenas nos dados de TREINO)
print("Aplicando SMOTE aos dados de Treinamento...")
sm = SMOTE(random_state=SEED) # Instancia o SMOTE.
# Aplica o SMOTE no conjunto de dados de treino normalizado.
X_res, y_res = sm.fit_resample(X_train_scaled, y_train_full) # Geração de amostras sintéticas.

print("-" * 60)
print("Distribuição de Classes no Treino DEPOIS do SMOTE:", Counter(y_res))
print(f"Número total de amostras no treino depois do SMOTE: {len(X_res)}")
print("-" * 60)

# 4.4 Configuração e Treinamento do Autoencoder (Autocloud)
INPUT_DIM = X_res.shape[1] # A dimensão de entrada é o número de dias/features.
model_ae_smote = create_autoencoder(INPUT_DIM)

print(f"Treinamento do Autoencoder com SMOTE (em {X_res.shape[0]} amostras balanceadas)...")

# Treinamento: Utiliza o dataset BALANCEADO (X_res) para aprender a reconstruir ambos os padrões.
# O X_res é usado como entrada e saída (auto-reconstrução).
model_ae_smote.fit(X_res, X_res,
                    epochs=50, # Usando 50 epochs conforme sua referência
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0)

print("Treinamento do Autoencoder (SMOTE) concluído.")

# 4.5 Avaliação
# Para calcular o limiar, o método mais robusto é usar SOMENTE os erros da classe NORMAL
# no conjunto de treino.

# 1. Obter o erro de reconstrução no conjunto de treino balanceado.
X_train_res_predictions = model_ae_smote.predict(X_res, verbose=0)
train_mse_res = np.mean(np.square(X_res - X_train_res_predictions), axis=1)

# 2. Obter os erros de reconstrução das amostras NORMAIS do treino (y_res == 0).
error_normal_train = train_mse_res[y_res == 0]

# 3. Definir o limiar (Threshold) com base nos dados normais do TREINO.
# O método estatístico é mais comum, mas usaremos o valor fixo da sua referência:
# threshold = np.mean(error_normal_train) + 2 * np.std(error_normal_train)
threshold = AE_THRESHOLD # Limiar fixo de 0.05 conforme sua referência.

print(f"\nLimiar (Threshold) Utilizado: {threshold:.4f} (Fixo/Referência)")
print("-" * 60)

# 4. Avaliar o modelo no conjunto de TESTE (X_test_scaled)
results_ae_smote = evaluate_autoencoder_model(model_ae_smote, X_test_scaled, y_test, threshold)
print_results("Autoencoder TREINADO com SMOTE - Resultados no Teste", results_ae_smote)
print("=" * 60)