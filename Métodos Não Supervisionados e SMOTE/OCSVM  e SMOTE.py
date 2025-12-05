# ==============================================================================
# SCRIPT 2/6: One-Class SVM (OCSVM) COM SMOTE
# Objetivo: Treinar e avaliar o OCSVM usando o SMOTE para balancear as classes
#           de treino antes de aplicar o modelo, tratando-o como um classificador
#           binário neste cenário (usando nu=0.5 no treino balanceado).
# Dependência: imblearn, scikit-learn
# ==============================================================================

# 1. Importações necessárias
import pandas as pd  # Linha 10: Manipulação de DataFrames.
import numpy as np  # Linha 11: Operações numéricas.
from collections import Counter  # Linha 12: Contagem de classes.
from imblearn.over_sampling import SMOTE  # Linha 13: Técnica de Sobreamostragem de Minoria Sintética.
from sklearn.preprocessing import MinMaxScaler  # Linha 14: Normalização.
from sklearn.model_selection import train_test_split  # Linha 15: Para separar dados de treino e teste.
from sklearn.svm import OneClassSVM  # Linha 16: O modelo de detecção de anomalias (OCSVM).
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score  # Linha 17: Métricas.
from scipy.interpolate import interp1d  # Linha 18: Para preenchimento de valores ausentes (interpolação).
import os # Linha 19

# --- Configurações Iniciais ---
SEED = 42 # Linha 22
np.random.seed(SEED) # Linha 23
os.environ['PYTHONHASHSEED'] = str(SEED) # Linha 24
DATA_PATH = os.path.join('.', 'data set.csv') # Linha 25
OCSVM_THRESHOLD = 0.0  # Limiar de decisão padrão (não usado no OCSVM, mas mantido por consistência). # Linha 26


# ==============================================================================
# 3. Funções de Suporte (Carregamento, Pré-processamento, Modelo e Avaliação)
# ==============================================================================

def load_sgcc_data(path): # Linha 33
    """Carrega o dataset real ou gera dados simulados se o arquivo não for encontrado."""
    try:
        df = pd.read_csv(path) # Linha 36
        TARGET_COLUMN = df.columns[-1] # Linha 37
        y = df[TARGET_COLUMN].values # Linha 38
        X_data_raw = df.drop(columns=[TARGET_COLUMN]) # Linha 39
        X_data = X_data_raw.apply(pd.to_numeric, errors='coerce').values # Linha 40
        N_DAYS = X_data.shape[1] # Linha 41
        print(f"Dados carregados: {X_data.shape[0]} consumidores, {N_DAYS} dias.") # Linha 42
        return X_data, y, N_DAYS # Linha 43
    except FileNotFoundError:
        # Lógica de SIMULAÇÃO caso o arquivo não seja encontrado
        N_SAMPLES = 1000; N_DAYS_SIM = 1035; FRAUD_RATE = 0.0853 # Linha 47
        time = np.linspace(0, 2 * np.pi, N_DAYS_SIM) # Linha 48
        X_normal = np.tile(50 + 20 * np.sin(time * 5), (int(N_SAMPLES * (1 - FRAUD_RATE)), 1)) + np.random.normal(0, 5, (int(N_SAMPLES * (1 - FRAUD_RATE)), N_DAYS_SIM)) # Linha 49
        X_fraud = np.tile(50 + 20 * np.sin(time * 5), (int(N_SAMPLES * FRAUD_RATE), 1)) # Linha 50
        for i in range(X_fraud.shape[0]): # Linha 51
            X_fraud[i, :np.random.randint(200, N_DAYS_SIM)] *= np.random.uniform(0.1, 0.5) # Linha 52
            X_fraud[i, :] += np.random.normal(0, 2, N_DAYS_SIM) # Linha 53
        X_sim = np.vstack([X_normal, X_fraud]); y_sim = np.array([0] * X_normal.shape[0] + [1] * X_fraud.shape[0]) # Linha 54
        print("ARQUIVO NÃO ENCONTRADO. Usando dados SIMULADOS.") # Linha 55
        return X_sim, y_sim, N_DAYS_SIM # Linha 56


def preprocess_data(X_data, fit_scaler=False, scaler=None): # Linha 59
    """Realiza interpolação linear, tratamento de outliers e normalização MinMaxScaler."""
    # Tratamento de valores ausentes (NaN) por interpolação linear
    X_imputed = X_data.copy() # Linha 62
    for i in range(X_imputed.shape[0]): # Linha 63
        series = X_imputed[i, :] # Linha 64
        not_nan_indices = np.where(~np.isnan(series))[0] # Linha 65
        nan_indices = np.where(np.isnan(series))[0] # Linha 66
        if len(not_nan_indices) >= 2: # Linha 67
            interp_func = interp1d(not_nan_indices, series[not_nan_indices], kind='linear', fill_value='extrapolate') # Linha 68
            series[nan_indices] = interp_func(nan_indices) # Linha 69
        series[np.isnan(series)] = 0 # Linha 70
        X_imputed[i, :] = series # Linha 71

    # Tratamento de Outliers (clipagem em Média + 2*Desvio Padrão)
    avg_x = np.mean(X_imputed, axis=0) # Linha 75
    std_x = np.std(X_imputed, axis=0) # Linha 76
    threshold_outlier = avg_x + 2 * std_x # Linha 77
    X_outlier_handled = X_imputed.copy() # Linha 78

    for j in range(X_outlier_handled.shape[1]): # Linha 80
        mask = X_outlier_handled[:, j] > threshold_outlier[j] # Linha 81
        X_outlier_handled[mask, j] = threshold_outlier[j] # Linha 82

    # Normalização (MinMaxScaler)
    if fit_scaler: # Linha 85
        scaler = MinMaxScaler() # Linha 86
        X_scaled = scaler.fit_transform(X_outlier_handled) # Linha 87
        return X_scaled, scaler # Linha 88
    else:
        X_scaled = scaler.transform(X_outlier_handled) # Linha 90
        return X_scaled, scaler # Linha 91


def evaluate_anomaly_model(model, X_test, y_test): # Linha 94: Função de avaliação detalhada (sem threshold, consistente com OCSVM)
    """
    Avalia o OCSVM e calcula as métricas detalhadas (por classe e Média Macro).
    O OCSVM retorna +1 (Normal) e -1 (Anomalia).
    """
    # Predição: +1 (Normal), -1 (Anomalia)
    y_pred_sklearn = model.predict(X_test) # Linha 100

    # 1. Mapeamento dos rótulos: -1 (Anomalia/Fraude) -> 1; +1 (Normal) -> 0
    y_pred = np.where(y_pred_sklearn == -1, 1, 0) # Linha 103

    # 2. Cálculo das Métricas (usando sklearn.metrics)

    # Métrica: Acurácia geral
    accuracy = accuracy_score(y_test, y_pred) * 100 # Linha 107

    # Métricas: Média Macro (average='macro')
    # A média macro trata igualmente ambas as classes, ideal para dados desbalanceados.
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100 # Linha 111
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100 # Linha 112
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100 # Linha 113

    # Métricas: Por Classe (average=None) -> [Valor Classe 0, Valor Classe 1]
    prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0) * 100 # Linha 116
    rec_per_class = recall_score(y_test, y_pred, average=None, zero_division=0) * 100 # Linha 117
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0) * 100 # Linha 118

    results = {
        'Acc': accuracy, # Linha 120
        'Prec(avg)': prec_macro, # Linha 121
        'Rec(avg)': rec_macro, # Linha 122
        'F1(avg)': f1_macro, # Linha 123
        # Resultados por classe
        'Prec(0)': prec_per_class[0], # Linha 125 (Classe 0: Normal)
        'Prec(1)': prec_per_class[1], # Linha 126 (Classe 1: Fraude)
        'Rec(0)': rec_per_class[0], # Linha 127
        'Rec(1)': rec_per_class[1], # Linha 128
        'F1(0)': f1_per_class[0], # Linha 129
        'F1(1)': f1_per_class[1], # Linha 130
    }
    return results # Linha 131


def print_results(title, results): # Linha 134
    """Imprime os resultados detalhados das métricas no formato de tabela."""
    print(f"\n--- {title} ---") # Linha 136
    print("Métrica | Classe 0 (Normal) | Classe 1 (Fraude) | Média Macro") # Linha 137 (Cabeçalho da Tabela)
    print("-" * 65) # Linha 138

    # Imprime Acurácia (é um valor único)
    acc = results['Acc'] # Linha 141
    print(f"Acc.    | {acc:.2f}% | {acc:.2f}% | {acc:.2f}%") # Linha 142

    # Imprime Precisão (Precision)
    prec_0, prec_1, prec_avg = results['Prec(0)'], results['Prec(1)'], results['Prec(avg)'] # Linha 145
    print(f"Prec.   | {prec_0:.2f}% | {prec_1:.2f}% | {prec_avg:.2f}%") # Linha 146

    # Imprime Recall
    rec_0, rec_1, rec_avg = results['Rec(0)'], results['Rec(1)'], results['Rec(avg)'] # Linha 149
    print(f"Recall  | {rec_0:.2f}% | {rec_1:.2f}% | {rec_avg:.2f}%") # Linha 150

    # Imprime F1-Score
    f1_0, f1_1, f1_avg = results['F1(0)'], results['F1(1)'], results['F1(avg)'] # Linha 153
    print(f"F1-Score| {f1_0:.2f}% | {f1_1:.2f}% | {f1_avg:.2f}%") # Linha 154
    print("-" * 65) # Linha 155


# ==============================================================================
# 4. Execução Principal: One-Class SVM TREINADO com SMOTE
# ==============================================================================

# 4.1 Carregar e Dividir Dados
X_raw, y, N_DAYS = load_sgcc_data(DATA_PATH) # Linha 161
# Separa 70% para treino e 30% para teste, mantendo a proporção de classes (stratify).
X_train_full, X_test, y_train_full, y_test = train_test_split(X_raw, y, test_size=0.3, random_state=SEED, stratify=y) # Linha 163

print("\n" + "=" * 60) # Linha 165
print(f"Dados de Treino (Original): {X_train_full.shape}") # Linha 166
print(f"Distribuição de Classes no Treino: {Counter(y_train_full)}") # Linha 167
print("=" * 60) # Linha 168

# 4.2 Pré-processamento e Normalização (Ajusta o scaler SOMENTE no treino)
X_train_scaled, scaler = preprocess_data(X_train_full, fit_scaler=True) # Linha 171
X_test_scaled, _ = preprocess_data(X_test, fit_scaler=False, scaler=scaler) # Linha 172

# 4.3 Aplicação do SMOTE (Apenas nos dados de TREINO)
print("Aplicando SMOTE aos dados de Treinamento...") # Linha 175
sm = SMOTE(random_state=SEED)  # Linha 176: Instancia o SMOTE.
# Aplica o SMOTE no conjunto de dados de treino normalizado.
X_res, y_res = sm.fit_resample(X_train_scaled, y_train_full)  # Linha 178: Geração de amostras sintéticas.

print("-" * 60) # Linha 180
print("Distribuição de Classes no Treino DEPOIS do SMOTE:", Counter(y_res)) # Linha 181
print(f"Número total de amostras no treino depois do SMOTE: {len(X_res)}") # Linha 182
print("-" * 60) # Linha 183

# 4.4 Configuração e Treinamento do OCSVM
OCSVM_PARAMS = { # Linha 186
    'kernel': 'rbf', # Linha 187
    # nu=0.5: Trata como um classificador binário, forçando 50% dos dados como outliers.
    'nu': 0.5, # Linha 191
    'gamma': 'auto', # Linha 192
} # Linha 193

print("\n--- OCSVM (SMOTE) - Configuração ---") # Linha 195
print("Parâmetros:", OCSVM_PARAMS) # Linha 196
print(f"Treinamento em {X_res.shape[0]} amostras BALANCEADAS.") # Linha 197

# Treinamento: Utiliza o dataset BALANCEADO (X_res, y_res)
model_ocsvm_smote = OneClassSVM(**OCSVM_PARAMS) # Linha 200
model_ocsvm_smote.fit(X_res, y_res) # Linha 201

print("Treinamento do OCSVM (SMOTE) concluído.") # Linha 203

# 4.5 Avaliação
# A avaliação é feita no conjunto de teste original (não sampleado)
results_ocsvm_smote = evaluate_anomaly_model(model_ocsvm_smote, X_test_scaled, y_test) # Linha 207: Chamada sem o parâmetro threshold
print_results("One-Class SVM TREINADO com SMOTE - Resultados Detalhados no Teste", results_ocsvm_smote) # Linha 208
print("=" * 60) # Linha 209
