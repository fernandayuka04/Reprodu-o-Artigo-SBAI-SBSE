import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings

# Ignora warnings futuros para manter o output limpo.
warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# 1. Leitura do Dataset da SGCC
# ATENÇÃO: Verifique se 'data set.csv' está no diretório correto.
# ==============================================================================
FILE_NAME = 'data set.csv'

try:
    print(f"1. Tentando ler o arquivo: {FILE_NAME}...")

    # Função pd.read_csv é a responsável por ler o arquivo CSV
    df = pd.read_csv(FILE_NAME)

    # Se a leitura for bem-sucedida, verifica as colunas principais
    if 'FLAG' not in df.columns:
        # Se a sua coluna target tiver outro nome (ex: 'ANOMALY'), mude aqui.
        raise ValueError("Coluna 'FLAG' (target) não encontrada no arquivo. Verifique o nome da sua coluna target.")

    print(f"Leitura bem-sucedida! Dimensões do dataset: {df.shape}")

except FileNotFoundError:
    print(f"ERRO: Arquivo '{FILE_NAME}' não encontrado. Gerando dados simulados como fallback.")

    # --- BLOCO DE FALLBACK (Geração de Dados Simulados) ---
    N_SAMPLES = 1000
    ANOMALY_RATIO = 0.05
    np.random.seed(42)
    data = {}
    for i in range(1, 31):
        data[f'C_Dia_{i}'] = np.random.normal(100, 15, N_SAMPLES)
    flags = np.zeros(N_SAMPLES, dtype=int)
    num_anomalies = int(N_SAMPLES * ANOMALY_RATIO)
    anomaly_indices = np.random.choice(N_SAMPLES, num_anomalies, replace=False)
    flags[anomaly_indices] = 1
    data['FLAG'] = flags
    df = pd.DataFrame(data)

    # Introduz valores ausentes artificialmente (CORRIGIDO O TYPEERROR AQUI)
    # 1. Seleciona 50 linhas aleatórias.
    row_indices = np.random.choice(df.index, size=50, replace=False)
    # 2. Seleciona 10 índices de coluna aleatórios (inteiros), exceto a última coluna ('FLAG')
    num_feature_cols = df.shape[1] - 1
    col_indices = np.random.choice(num_feature_cols, size=10, replace=False)

    # Usa .iloc com índices inteiros, resolvendo o TypeError
    df.iloc[row_indices, col_indices] = np.nan
    # ----------------------------------------------------

print(f"Contagem de classes antes do pré-processamento:\n{df['FLAG'].value_counts()}")

# ==============================================================================
# 2. Pré-processamento: Interpolação Linear para Valores Ausentes
# Esta etapa replica o tratamento de 'missing values' (valores ausentes) do artigo.
# ==============================================================================
print("\n2. Aplicando Interpolação Linear para valores ausentes (anomalias e perdas não técnicas)...")

# O método 'interpolate(method='linear', axis=0)' preenche os NaNs
# usando valores lineares entre os pontos conhecidos.
# É aplicado em todo o DataFrame de features (X) antes da divisão treino/teste.
X = df.drop('FLAG', axis=1)
Y = df['FLAG']
X_interpolated = X.interpolate(method='linear', axis=0)

# ==============================================================================
# 3. Feature Engineering: Aplicação da Fórmula (Índice de Amplitude de Consumo)
# Esta é a etapa que você solicitou para aplicar uma fórmula.
# ==============================================================================
print("\n3. Criando a feature 'Índice de Amplitude de Consumo' (IAC) pela fórmula...")

# A Fórmula utilizada (IAC): (Max Consumo - Min Consumo) / Média Consumo
# O objetivo é capturar a volatilidade do consumo, um forte indicador de roubo de energia.

# Parâmetros da Fórmula:
# .max(axis=1): Máximo consumo em 30 dias por cliente (linha).
# .min(axis=1): Mínimo consumo em 30 dias por cliente (linha).
# .mean(axis=1): Média de consumo em 30 dias por cliente (linha).

# Cria a nova coluna 'IAC'
max_consumption = X_interpolated.max(axis=1)
min_consumption = X_interpolated.min(axis=1)
mean_consumption = X_interpolated.mean(axis=1)

# Cálculo da fórmula:
# Usa 'np.where' para evitar divisão por zero, caso a média seja 0.
X_interpolated['IAC'] = np.where(
    mean_consumption != 0,
    (max_consumption - min_consumption) / mean_consumption,
    0
)

# ==============================================================================
# 4. Divisão e Aplicação do SMOTE (Balanceamento de Classes)
# O SMOTE é aplicado APENAS no conjunto de treinamento para evitar data leakage.
# ==============================================================================
# Divisão dos Dados (80% treino, 20% teste)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_interpolated, Y, test_size=0.2, random_state=42, stratify=Y
)

print("\n4. Aplicando SMOTE no conjunto de treino para balanceamento...")
smote = SMOTE(random_state=42)

# Aplica o SMOTE: gera dados sintéticos da classe minoritária (FLAG=1)
# até que as classes estejam balanceadas.
X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)

print(f"Contagem de classes do TREINO ORIGINAL:\n{Y_train.value_counts()}")
print(f"Contagem de classes do TREINO COM SMOTE (agora igualadas):\n{Y_train_res.value_counts()}")

# ==============================================================================
# 5. Criação do Novo Arquivo CSV com Dados Balanceados
# O usuário solicitou um novo arquivo com os dados interpolados e balanceados.
# Salvamos o conjunto de treinamento balanceado.
# ==============================================================================
# Junta as features balanceadas (X_train_res) e o target balanceado (Y_train_res)
df_resampled = pd.DataFrame(X_train_res, columns=X_train.columns)
df_resampled['FLAG'] = Y_train_res

output_filename = 'sgcc_smote_data.csv'
df_resampled.to_csv(output_filename, index=False)

print(f"\nPré-processamento concluído!")
print(f"Arquivo de saída criado: '{output_filename}' com {len(df_resampled)} amostras.")
print(f"Amostras Normal (0) e Anomalia (1) estão balanceadas neste novo arquivo.")