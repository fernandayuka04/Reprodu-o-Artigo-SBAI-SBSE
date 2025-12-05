import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import classification_report, roc_auc_score


# ==============================================================================
# 1. Geração de Dados Simulados (NORMAL e ANÔMALO)
# ==============================================================================

def generate_simulated_data(n_normal=10000, n_anomaly=200):
    """Gera dados simulados de séries temporais para o exemplo de AutoEncoder."""

    N_FEATURES = 30  # Simula 30 dias de consumo
    np.random.seed(42)

    # --- 1.1 Geração de Dados NORMAIS ---
    # Consumo típico: Distribuição normal com média 100 e desvio padrão 15
    normal_data = np.random.normal(loc=100, scale=15, size=(n_normal, N_FEATURES))
    # Cria rótulos (FLAG) para dados normais: 0
    normal_labels = np.zeros(n_normal, dtype=int)

    # --- 1.2 Geração de Dados ANÔMALOS (Simulação de Fraude) ---
    # Fraude/Anomalia: Distribuição com média e/ou variância muito diferentes
    # Aqui, simulamos um consumo anormalmente alto e volátil.
    anomaly_data = np.random.normal(loc=150, scale=30, size=(n_anomaly, N_FEATURES))
    # Cria rótulos (FLAG) para dados anômalos: 1
    anomaly_labels = np.ones(n_anomaly, dtype=int)

    # --- 1.3 Combinação e DataFrame ---
    # Combina dados e rótulos
    X_all = np.vstack([normal_data, anomaly_data])
    Y_all = np.concatenate([normal_labels, anomaly_labels])

    # Cria o DataFrame
    df = pd.DataFrame(X_all, columns=[f'Dia_{i + 1}' for i in range(N_FEATURES)])
    df['FLAG'] = Y_all

    print(f"Dados simulados gerados: {len(df)} amostras (Classes: {df['FLAG'].value_counts().to_dict()})")
    return df


# ==============================================================================
# 2. Pré-processamento e Filtragem (Crucial para AutoEncoder)
# ==============================================================================

def preprocess_and_split(df):
    """Separa, escala e filtra o conjunto de treino para conter APENAS dados normais."""

    X = df.drop('FLAG', axis=1)
    Y = df['FLAG']

    # Divisão treino-teste (80% treino, 20% teste)
    X_train_full, X_test, Y_train_full, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    # PASSO CRÍTICO: AutoEncoder só treina em dados NORMAIS (FLAG=0)
    X_train_normal = X_train_full[Y_train_full == 0]

    # Padronização (Scaling): Otimiza o treinamento da Rede Neural
    scaler = StandardScaler()

    # Ajusta o scaler APENAS nos dados normais de treinamento
    X_train_scaled = scaler.fit_transform(X_train_normal)

    # Transforma o conjunto de teste (contém normais e anomalias)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, Y_test


# ==============================================================================
# 3. Definição do Modelo AutoEncoder Genérico
# ==============================================================================
def create_autoencoder(input_dim, encoding_dim=8):
    """Define a arquitetura de uma rede AutoEncoder genérica."""

    # Define a camada de entrada (dimensão = número de features)
    input_layer = Input(shape=(input_dim,))

    # --- ENCODER (Comprime) ---
    encoder = Dense(int(input_dim * 0.7), activation="relu")(input_layer)
    # Camada de gargalo (bottleneck): Representação compacta
    encoder = Dense(encoding_dim, activation="relu", name="bottleneck")(encoder)

    # --- DECODER (Reconstrói) ---
    decoder = Dense(int(input_dim * 0.7), activation='relu')(encoder)
    # Camada de saída: Deve ter a mesma dimensão da entrada
    output_layer = Dense(input_dim, activation='linear')(decoder)

    # Cria o modelo
    model = Model(inputs=input_layer, outputs=output_layer, name="Generic_AutoEncoder")

    # Compilação: Usa MSE (Erro Quadrático Médio) como perda
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

    return model


# ==============================================================================
# 4. Execução Principal do AutoEncoder
# ==============================================================================

print("--- 1. Geração e Preparação de Dados ---")
df_simulated = generate_simulated_data()
X_train_scaled, X_test_scaled, Y_test = preprocess_and_split(df_simulated)

INPUT_DIM = X_train_scaled.shape[1]
ENCODING_DIM = 8  # Dimensão do gargalo (pode ser ajustada)

print(f"\nDimensão dos dados (Input): {INPUT_DIM}")
print(f"Dimensão do gargalo (Bottleneck): {ENCODING_DIM}")

# Cria o modelo
autoencoder = create_autoencoder(INPUT_DIM, ENCODING_DIM)
autoencoder.summary()

print("\n--- 2. Treinamento (Apenas Dados Normais) ---")
# Treina o modelo para reconstruir a própria entrada
history = autoencoder.fit(
    X_train_scaled, X_train_scaled,
    epochs=50,
    batch_size=64,
    shuffle=True,
    validation_data=(X_test_scaled, X_test_scaled),
    verbose=0
)

# ==============================================================================
# 5. Avaliação e Detecção de Anomalias
# ==============================================================================

print("\n--- 3. Detecção de Anomalias (Cálculo do Erro) ---")

# 5.1. Calcular o Erro de Reconstrução no Teste
predictions = autoencoder.predict(X_test_scaled, verbose=0)
# MSE: Erro entre o original e o reconstruído
mse = np.mean(np.power(X_test_scaled - predictions, 2), axis=1)

error_df = pd.DataFrame({'Reconstruction_Error': mse, 'True_Class': Y_test})

# 5.2. Determinação do Limiar (Threshold)
# O limiar é o 95º percentil dos erros de reconstrução da classe NORMAL (FLAG=0)
normal_error = error_df[error_df['True_Class'] == 0]['Reconstruction_Error']
THRESHOLD = np.percentile(normal_error, 95)

print(f"Limiar de Anomalia (95º Percentil do Erro Normal): {THRESHOLD:.4f}")

# 5.3. Classificação e Resultados
# Classifica como 1 (Anomalia) se o erro for maior que o limiar
Y_pred = (error_df['Reconstruction_Error'] > THRESHOLD).astype(int)

print("\n" + "=" * 50)
print("RESULTADO GENÉRICO DA DETECÇÃO DE ANOMALIAS (AutoEncoder)")
print("=" * 50)

# AUC-ROC usando o erro como pontuação
auc_score = roc_auc_score(Y_test, mse)
print(f"AUC-ROC (Pontuação de Erro de Reconstrução): {auc_score:.4f}")

# Relatório de Classificação
print("\nRelatório de Classificação (Anomalias = Classe 1):")
print(classification_report(Y_test, Y_pred, target_names=['Normal (0)', 'Anomalia (1)']))