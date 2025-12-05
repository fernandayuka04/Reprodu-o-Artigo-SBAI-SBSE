import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import classification_report, roc_auc_score

# --- CONFIGURAÇÃO ---
# Nome do arquivo de dados SGCC fornecido pelo usuário
# O caminho está correto com o prefixo 'r'
DATA_FILE = r'C:\Users\econt\Reprodução_Experimento\data set\data set.csv'
ENCODING_DIM = 10  # Tamanho do gargalo (bottleneck) para o AutoEncoder/AutoCloud


# ==============================================================================
# 1. Carregamento dos Dados
# ==============================================================================

def load_sgcc_data(file_path):
    """Carrega os dados SGCC do arquivo CSV e garante que as colunas de consumo sejam numéricas."""
    try:
        df = pd.read_csv(file_path)
        print(f"Dados carregados com sucesso de: {file_path}")

        # 1. Validação de Colunas Críticas
        if 'FLAG' not in df.columns:
            print("\nERRO CRÍTICO: A coluna 'FLAG' (rótulo de fraude) não foi encontrada no arquivo CSV.")
            return None

        # Colunas a serem excluídas da conversão: 'FLAG' e, se presente, 'CONS_NO'
        cols_to_exclude = ['FLAG']
        if 'CONS_NO' in df.columns:
            cols_to_exclude.append('CONS_NO')

        time_series_cols = df.columns.drop(cols_to_exclude, errors='ignore').tolist()

        # 2. Conversão Explícita de Tipo de Dados (Solução para o TypeError)
        # Força as colunas de consumo a serem numéricas. 'coerce' converte qualquer valor
        # não numérico (como strings vazias ou caracteres especiais) para NaN.
        print(f"\nCoerção de {len(time_series_cols)} colunas de consumo para tipo numérico (float)...")
        # Usamos .apply(pd.to_numeric) que é mais robusto para DataFrames
        df[time_series_cols] = df[time_series_cols].apply(pd.to_numeric, errors='coerce')

        # 3. Contagem e Visualização
        class_counts = df['FLAG'].value_counts().to_dict()
        print(f"Amostras totais: {len(df)} (Classes: {class_counts})")
        print(f"Colunas do dataset: {df.columns.tolist()}")

        return df

    except FileNotFoundError:
        print(f"\nERRO: Arquivo não encontrado em '{file_path}'. Verifique o caminho e o nome do arquivo.")
        return None
    except Exception as e:
        print(f"\nOcorreu um erro ao carregar/converter os dados: {e}")
        return None


# ==============================================================================
# 2. Pré-processamento SGCC (Interpolação e Engenharia de Features)
# ==============================================================================

def preprocess_sgcc_data(df):
    """Aplica Interpolação Linear e Engenharia de Features como no método AutoCloud/SGCC."""

    print("\n--- Aplicando Pré-processamento SGCC (Interpolação Linear e Feature Engineering) ---")

    # Colunas de identificação que não são séries temporais, mas que devem ser excluídas do cálculo
    non_ts_cols = ['FLAG']
    if 'CONS_NO' in df.columns:
        non_ts_cols.append('CONS_NO')

    # Identifica as colunas de série temporal (consumo)
    time_series_cols = df.columns.drop(non_ts_cols, errors='ignore').tolist()

    # DataFrame contendo APENAS os valores de consumo (agora garantidamente numéricos)
    df_ts = df[time_series_cols].copy()

    # 2.1. Interpolação Linear para Tratar Missing Values (MVs)
    # Interpolação ao longo da linha (axis=1) para preencher MVs na série temporal de cada consumidor
    df_interp = df_ts.interpolate(method='linear', axis=1)

    # Preenche MVs restantes (se houver, geralmente no início ou fim da série) com a média
    df_interp = df_interp.fillna(df_interp.mean())

    # 2.2. Engenharia de Features (Simulando o Índice de Amplitude, etc.)
    # Cria novas features que capturam estatísticas importantes do perfil de consumo
    df_interp['Mean'] = df_interp.mean(axis=1)
    df_interp['StdDev'] = df_interp.std(axis=1)
    df_interp['Amplitude'] = df_interp.max(axis=1) - df_interp.min(axis=1)

    # 2.3. Junta Colunas de Identificação (se existirem) e a FLAG de volta
    if 'CONS_NO' in df.columns:
        df_interp['CONS_NO'] = df['CONS_NO']

    df_interp['FLAG'] = df['FLAG']

    print(f"Número total de features após engenharia: {df_interp.shape[1] - (2 if 'CONS_NO' in df.columns else 1)}")
    print(
        f"Colunas finais de features: {df_interp.columns.drop(['FLAG', 'CONS_NO'], errors='ignore').tolist()[:5]}... (+ {len(df_interp.columns) - 7 if 'CONS_NO' in df.columns else len(df_interp.columns) - 4} outras)")
    print("Pré-processamento concluído.")
    return df_interp


# ==============================================================================
# 3. Divisão, Filtragem (Apenas Normal) e Padronização
# ==============================================================================

def split_filter_and_scale(df):
    """Divide, filtra para treino (apenas normal) e padroniza os dados."""

    # Remove CONS_NO se presente, pois não é um feature de treino (é apenas um ID)
    features_to_drop = ['FLAG']
    if 'CONS_NO' in df.columns:
        features_to_drop.append('CONS_NO')

    X = df.drop(features_to_drop, axis=1)
    Y = df['FLAG']

    # Divisão treino-teste (20% para teste)
    X_train_full, X_test, Y_train_full, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    # PASSO CRÍTICO DO AUTOCLOUD/AUTOENCODER: Treinar APENAS em dados NORMAIS (FLAG=0)
    X_train_normal = X_train_full[Y_train_full == 0]

    # Padronização (StandardScaler)
    scaler = StandardScaler()

    # Ajusta o scaler APENAS nos dados normais de treinamento
    X_train_scaled = scaler.fit_transform(X_train_normal)

    # Transforma o conjunto de teste (contém normais e anomalias)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTamanho do conjunto de Treino (somente Normal): {X_train_scaled.shape}")
    print(f"Tamanho do conjunto de Teste (Normal e Anomalia): {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, Y_test


# ==============================================================================
# 4. Definição do Modelo AutoEncoder (A arquitetura do AutoCloud)
# ==============================================================================
def create_autoencoder(input_dim, encoding_dim):
    """Define a arquitetura do AutoEncoder/AutoCloud."""

    # 1. ENCODER (Comprime)
    input_layer = Input(shape=(input_dim,))

    # Camadas do Encoder
    encoder = Dense(int(input_dim * 0.7), activation="relu")(input_layer)
    encoder = Dense(int(input_dim * 0.5), activation="relu")(encoder)

    # Camada de gargalo (bottleneck)
    bottleneck = Dense(encoding_dim, activation="relu", name="bottleneck")(encoder)

    # 2. DECODER (Reconstrói)
    # Camadas do Decoder
    decoder = Dense(int(input_dim * 0.5), activation='relu')(bottleneck)
    decoder = Dense(int(input_dim * 0.7), activation='relu')(decoder)

    # Camada de saída (mesma dimensão da entrada)
    output_layer = Dense(input_dim, activation='linear')(decoder)

    # Cria o modelo
    model = Model(inputs=input_layer, outputs=output_layer, name="SGCC_AutoCloud")

    # Compilação: Usa MSE (Erro Quadrático Médio) como perda
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

    return model


# ==============================================================================
# 5. Execução Principal e Avaliação
# ==============================================================================

# --- A. Preparação dos Dados ---
df_raw = load_sgcc_data(DATA_FILE)

if df_raw is None:
    print("Execução interrompida devido a um erro no carregamento dos dados.")
else:
    df_processed = preprocess_sgcc_data(df_raw.copy())

    X_train_scaled, X_test_scaled, Y_test = split_filter_and_scale(df_processed)

    INPUT_DIM = X_train_scaled.shape[1]

    # --- B. Criação e Treinamento do Modelo ---
    autoencoder = create_autoencoder(INPUT_DIM, ENCODING_DIM)

    # A arquitetura do AutoEncoder (o "Cloud")
    autoencoder.summary()

    print("\n--- 4. Treinamento do AutoCloud (Apenas Dados Normais) ---")

    # O AutoCloud treina para minimizar o erro de reconstrução dos dados normais
    history = autoencoder.fit(
        X_train_scaled, X_train_scaled,
        epochs=100,  # Ajuste conforme a necessidade
        batch_size=64,
        shuffle=True,
        validation_data=(X_test_scaled, X_test_scaled),
        verbose=1  # Mostrar progresso do treinamento
    )

    # --- C. Detecção de Anomalias e Avaliação ---
    print("\n--- 5. Detecção de Anomalias e Avaliação de Fraude (AutoCloud) ---")

    # 5.1. Calcular o Erro de Reconstrução no Teste
    predictions = autoencoder.predict(X_test_scaled, verbose=0)
    # MSE: Erro entre o original e o reconstruído
    mse = np.mean(np.power(X_test_scaled - predictions, 2), axis=1)

    error_df = pd.DataFrame({'Reconstruction_Error': mse, 'True_Class': Y_test})

    # 5.2. Determinação do Limiar (Threshold)
    # Limiar: 95º percentil dos erros de reconstrução da classe NORMAL no conjunto de teste
    normal_error = error_df[error_df['True_Class'] == 0]['Reconstruction_Error']
    THRESHOLD = np.percentile(normal_error, 95)

    print(f"Limiar de Anomalia (95º Percentil do Erro Normal): {THRESHOLD:.4f}")

    # 5.3. Classificação e Resultados
    # Classifica como 1 (Anomalia/Fraude) se o erro for maior que o limiar
    Y_pred = (error_df['Reconstruction_Error'] > THRESHOLD).astype(int)

    print("\n" + "=" * 60)
    print("RESULTADO DO MÉTODO AUTOCLOUD (AUTOENCODER - DADOS DO CSV)")
    print("=" * 60)

    # AUC-ROC usando o erro como pontuação
    auc_score = roc_auc_score(Y_test, mse)
    print(f"AUC-ROC (Pontuação de Erro de Reconstrução): {auc_score:.4f}")

    # Relatório de Classificação (O foco é o Recall na classe 1 - Anomalia/Fraude)
    print("\nRelatório de Classificação (Anomalias = Classe 1):")
    print(classification_report(Y_test, Y_pred, target_names=['Normal (0)', 'Anomalia (1)']))