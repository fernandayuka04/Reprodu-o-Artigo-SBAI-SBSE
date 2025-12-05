# ==============================================================================
# SCRIPT 4/6: Isolation Forest (iForest) COM SMOTE
# Objetivo: Treinar e avaliar o iForest usando o SMOTE para balancear as classes
#           de treino antes de aplicar o modelo.
# Dependência: imblearn, scikit-learn
# ==============================================================================

# 1. Importações necessárias
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
import tensorflow as tf # Para configurar a seed (mantido por consistência de SEED)

# 2. Configurações Iniciais
SEED = 42
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(SEED)
# Simulação do caminho do arquivo de dados
DATA_PATH = os.path.join('.', 'data set.csv')


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


def evaluate_anomaly_model(model, X_test, y_test):
    """
    Avalia o iForest e calcula as métricas detalhadas (por classe e Média Macro).
    O iForest retorna +1 (Normal) e -1 (Anomalia).
    """
    # Predição: +1 (Normal), -1 (Anomalia)
    y_pred_sklearn = model.predict(X_test)

    # 1. Mapeamento dos rótulos: -1 (Anomalia/Fraude) -> 1; +1 (Normal) -> 0
    # O modelo agora está em conformidade com os rótulos 0 e 1 do y_test.
    y_pred = np.where(y_pred_sklearn == -1, 1, 0) # Linha 127

    # 2. Cálculo das Métricas (usando sklearn.metrics)

    # Métrica: Acurácia geral
    accuracy = accuracy_score(y_test, y_pred) * 100 # Linha 130

    # Métricas: Média Macro (Prec(avg), Rec(avg), F1(avg)) - Corresponde à coluna Média
    # average='macro' calcula a média não ponderada das classes (média relevante para desbalanceamento)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100 # Linha 134
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100     # Linha 135
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100          # Linha 136

    # Métricas: Por Classe (average=None) -> Permite determinar Maior/Menor
    # average=None retorna um array: [Valor Classe 0, Valor Classe 1]
    prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0) * 100 # Linha 140
    rec_per_class = recall_score(y_test, y_pred, average=None, zero_division=0) * 100     # Linha 141
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0) * 100          # Linha 142

    results = {
        'Acc': accuracy,     # Linha 145
        'Prec(avg)': prec_macro, # Linha 146
        'Rec(avg)': rec_macro,   # Linha 147
        'F1(avg)': f1_macro,     # Linha 148
        # Resultados por classe (usados para preencher as colunas Maior e Menor)
        'Prec(0)': prec_per_class[0], # Linha 150
        'Prec(1)': prec_per_class[1], # Linha 151
        'Rec(0)': rec_per_class[0],   # Linha 152
        'Rec(1)': rec_per_class[1],   # Linha 153
        'F1(0)': f1_per_class[0],     # Linha 154
        'F1(1)': f1_per_class[1],     # Linha 155
    }
    return results


def print_results(title, results):
    """Imprime os resultados detalhados das métricas no formato de tabela."""
    print(f"\n--- {title} ---") # Linha 161
    print("Métrica | Classe 0 (Normal) | Classe 1 (Fraude) | Média Macro") # Linha 162 (Cabeçalho)
    print("-" * 65) # Linha 163

    # Imprime Acurácia (é um valor único)
    acc = results['Acc']
    print(f"Acc.    | {acc:.2f}% | {acc:.2f}% | {acc:.2f}%") # Linha 167

    # Imprime Precisão (Precision)
    prec_0, prec_1, prec_avg = results['Prec(0)'], results['Prec(1)'], results['Prec(avg)']
    print(f"Prec.   | {prec_0:.2f}% | {prec_1:.2f}% | {prec_avg:.2f}%") # Linha 171

    # Imprime Recall
    rec_0, rec_1, rec_avg = results['Rec(0)'], results['Rec(1)'], results['Rec(avg)']
    print(f"Recall  | {rec_0:.2f}% | {rec_1:.2f}% | {rec_avg:.2f}%") # Linha 175

    # Imprime F1-Score
    f1_0, f1_1, f1_avg = results['F1(0)'], results['F1(1)'], results['F1(avg)']
    print(f"F1-Score| {f1_0:.2f}% | {f1_1:.2f}% | {f1_avg:.2f}%") # Linha 179
    print("-" * 65) # Linha 180


# ==============================================================================
# 4. Execução Principal: Isolation Forest TREINADO com SMOTE
# ==============================================================================

# 4.1 Carregar e Dividir Dados
X_raw, y, N_DAYS = load_sgcc_data(DATA_PATH)  # Carrega os dados (reais ou simulados). # Linha 186
# Separa 70% para treino e 30% para teste, mantendo a proporção de classes (stratify).
X_train_full, X_test, y_train_full, y_test = train_test_split(X_raw, y, test_size=0.3, random_state=SEED, stratify=y) # Linha 188

print("\n" + "=" * 60) # Linha 190
print(f"Dados de Treino (Original): {X_train_full.shape}") # Linha 191
print(f"Distribuição de Classes no Treino: {Counter(y_train_full)}") # Linha 192
print("=" * 60) # Linha 193

# 4.2 Pré-processamento e Normalização (Ajusta o scaler SOMENTE no treino)
X_train_scaled, scaler = preprocess_data(X_train_full, fit_scaler=True) # Linha 196
X_test_scaled, _ = preprocess_data(X_test, fit_scaler=False, scaler=scaler) # Linha 197

# 4.3 Aplicação do SMOTE (Apenas nos dados de TREINO)
print("Aplicando SMOTE aos dados de Treinamento...") # Linha 200
sm = SMOTE(random_state=SEED)  # Instancia o SMOTE. # Linha 201
# Aplica o SMOTE no conjunto de dados de treino normalizado.
X_res, y_res = sm.fit_resample(X_train_scaled, y_train_full)  # Geração de amostras sintéticas. # Linha 203

print("-" * 60) # Linha 205
print("Distribuição de Classes no Treino DEPOIS do SMOTE:", Counter(y_res)) # Linha 206
print(f"Número total de amostras no treino depois do SMOTE: {len(X_res)}") # Linha 207
print("-" * 60) # Linha 208

# 4.4 Configuração e Treinamento do Isolation Forest
# Parâmetros de referência:
IFOREST_PARAMS = {
    'n_estimators': 100,
    'max_samples': 'auto',
    # Como estamos treinando em dados BALANCEADOS (50/50), o contamination deve ser 0.5.
    'contamination': 0.5,
    'random_state': SEED
} # Linhas 211-218

print("\n--- iForest (SMOTE) - Configuração ---") # Linha 220
print("Parâmetros:", IFOREST_PARAMS) # Linha 221
print(f"Treinamento em {X_res.shape[0]} amostras BALANCEADAS.") # Linha 222

# Treinamento: Utiliza o dataset BALANCEADO (X_res, y_res)
model_iforest_smote = IsolationForest(**IFOREST_PARAMS) # Linha 225
model_iforest_smote.fit(X_res, y_res) # Linha 226

print("Treinamento do iForest (SMOTE) concluído.") # Linha 228

# 4.5 Avaliação
# A chamada foi ajustada para a nova assinatura da função.
results_iforest_smote = evaluate_anomaly_model(model_iforest_smote, X_test_scaled, y_test) # Linha 232
print_results("Isolation Forest TREINADO com SMOTE - Resultados Detalhados no Teste", results_iforest_smote) # Linha 233
print("=" * 60) # Linha 234
