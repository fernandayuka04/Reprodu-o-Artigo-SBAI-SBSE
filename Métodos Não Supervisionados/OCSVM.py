# ==============================================================================
# SCRIPT 1/6: One-Class SVM (OCSVM) SEM SMOTE
# Objetivo: Treinar e avaliar o OCSVM usando apenas a classe normal (0)
#           sem aumentar o número de amostras (None Sampling).
# Dependência: scikit-learn
# ==============================================================================

# 1. Importações necessárias
import numpy as np # Linha 9
import pandas as pd # Linha 10
from collections import Counter # Linha 11: Para contagem de classes no terminal
from sklearn.model_selection import train_test_split # Linha 12
from sklearn.preprocessing import MinMaxScaler # Linha 13
from sklearn.svm import OneClassSVM # Linha 14: O modelo de detecção de anomalias
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score # Linha 15: Métricas de avaliação
from scipy.interpolate import interp1d # Linha 16
import os # Linha 17

# --- Configurações Iniciais ---
SEED = 42 # Linha 20
np.random.seed(SEED) # Linha 21
os.environ['PYTHONHASHSEED'] = str(SEED) # Linha 22
DATA_PATH = os.path.join('.', 'data set.csv') # Linha 23: Corrigi o caminho para o padrão local (uso do '.')

# ==============================================================================
# 3. Funções de Suporte (Carregamento, Pré-processamento, Modelo e Avaliação)
# ==============================================================================

def load_sgcc_data(path): # Linha 29
    """Carrega o dataset real ou gera dados simulados se o arquivo não for encontrado."""
    try:
        df = pd.read_csv(path) # Linha 32
        TARGET_COLUMN = df.columns[-1] # Linha 33
        y = df[TARGET_COLUMN].values # Linha 34
        X_data_raw = df.drop(columns=[TARGET_COLUMN]) # Linha 35
        # Converte para numpy array
        X_data = X_data_raw.apply(pd.to_numeric, errors='coerce').values # Linha 38
        N_DAYS = X_data.shape[1] # Linha 39
        print(f"Dados carregados: {X_data.shape[0]} consumidores, {N_DAYS} dias.") # Linha 40
        return X_data, y, N_DAYS # Linha 41
    except FileNotFoundError:
        # Lógica de SIMULAÇÃO caso o arquivo não seja encontrado
        N_SAMPLES = 1000; N_DAYS_SIM = 1035; FRAUD_RATE = 0.0853 # Linha 45
        time = np.linspace(0, 2 * np.pi, N_DAYS_SIM) # Linha 46
        X_normal = np.tile(50 + 20 * np.sin(time * 5), (int(N_SAMPLES * (1 - FRAUD_RATE)), 1)) + np.random.normal(0, 5, (int(N_SAMPLES * (1 - FRAUD_RATE)), N_DAYS_SIM)) # Linha 47
        X_fraud = np.tile(50 + 20 * np.sin(time * 5), (int(N_SAMPLES * FRAUD_RATE), 1)) # Linha 48
        for i in range(X_fraud.shape[0]): # Linha 49
            X_fraud[i, :np.random.randint(200, N_DAYS_SIM)] *= np.random.uniform(0.1, 0.5) # Linha 50
            X_fraud[i, :] += np.random.normal(0, 2, N_DAYS_SIM) # Linha 51
        X_sim = np.vstack([X_normal, X_fraud]); y_sim = np.array([0] * X_normal.shape[0] + [1] * X_fraud.shape[0]) # Linha 52
        print("ERRO: Arquivo não encontrado. Usando dados SIMULADOS.") # Linha 53
        return X_sim, y_sim, N_DAYS_SIM # Linha 54

def preprocess_data(X_data, fit_scaler=False, scaler=None): # Linha 57
    """Imputação (Interpolação), Tratamento de Outliers (2-Sigma) e Normalização."""
    X_imputed = X_data.copy() # Linha 59
    for i in range(X_imputed.shape[0]): # Linha 60
        series = X_imputed[i, :] # Linha 61
        not_nan_indices = np.where(~np.isnan(series))[0] # Linha 62
        nan_indices = np.where(np.isnan(series))[0] # Linha 63
        if len(not_nan_indices) >= 2: # Linha 64
            interp_func = interp1d(not_nan_indices, series[not_nan_indices], kind='linear', fill_value='extrapolate') # Linha 65
            series[nan_indices] = interp_func(nan_indices) # Linha 66
        series[np.isnan(series)] = 0 # Linha 67
        X_imputed[i, :] = series # Linha 68

    # Tratamento de Outliers (2-Sigma)
    avg_x = np.mean(X_imputed, axis=0) # Linha 72
    std_x = np.std(X_imputed, axis=0) # Linha 73
    threshold = avg_x + 2 * std_x # Linha 74
    X_outlier_handled = X_imputed.copy() # Linha 75

    for j in range(X_outlier_handled.shape[1]): # Linha 77
        mask = X_outlier_handled[:, j] > threshold[j] # Linha 78
        X_outlier_handled[mask, j] = threshold[j] # Linha 79

    # Normalização
    if fit_scaler: # Linha 82
        scaler = MinMaxScaler() # Linha 83
        X_scaled = scaler.fit_transform(X_outlier_handled) # Linha 84
        return X_scaled, scaler # Linha 85
    else:
        X_scaled = scaler.transform(X_outlier_handled) # Linha 87
        return X_scaled, scaler # Linha 88


def evaluate_anomaly_model(model, X_test, y_test): # Linha 91: Função de avaliação (assinatura simplificada para consistência)
    """
    Avalia o OCSVM e calcula as métricas detalhadas (por classe e Média Macro).
    O OCSVM retorna +1 (Normal) e -1 (Anomalia).
    """
    # Predição: +1 (Normal), -1 (Anomalia)
    y_pred_sklearn = model.predict(X_test) # Linha 97

    # 1. Mapeamento dos rótulos: -1 (Anomalia/Fraude) -> 1; +1 (Normal) -> 0
    y_pred = np.where(y_pred_sklearn == -1, 1, 0) # Linha 100

    # 2. Cálculo das Métricas (usando sklearn.metrics)

    # Métrica: Acurácia geral
    accuracy = accuracy_score(y_test, y_pred) * 100 # Linha 104

    # Métricas: Média Macro (average='macro')
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100 # Linha 107
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100 # Linha 108
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100 # Linha 109

    # Métricas: Por Classe (average=None) -> [Valor Classe 0, Valor Classe 1]
    prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0) * 100 # Linha 112
    rec_per_class = recall_score(y_test, y_pred, average=None, zero_division=0) * 100 # Linha 113
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0) * 100 # Linha 114

    results = {
        'Acc': accuracy, # Linha 116
        'Prec(avg)': prec_macro, # Linha 117
        'Rec(avg)': rec_macro, # Linha 118
        'F1(avg)': f1_macro, # Linha 119
        # Resultados por classe
        'Prec(0)': prec_per_class[0], # Linha 121
        'Prec(1)': prec_per_class[1], # Linha 122
        'Rec(0)': rec_per_class[0], # Linha 123
        'Rec(1)': rec_per_class[1], # Linha 124
        'F1(0)': f1_per_class[0], # Linha 125
        'F1(1)': f1_per_class[1], # Linha 126
    }
    return results # Linha 127


def print_results(title, results): # Linha 130
    """Imprime os resultados detalhados das métricas no formato de tabela."""
    print(f"\n--- {title} ---") # Linha 132
    print("Métrica | Classe 0 (Normal) | Classe 1 (Fraude) | Média Macro") # Linha 133 (Cabeçalho)
    print("-" * 65) # Linha 134

    # Imprime Acurácia (é um valor único)
    acc = results['Acc'] # Linha 137
    print(f"Acc.    | {acc:.2f}% | {acc:.2f}% | {acc:.2f}%") # Linha 138

    # Imprime Precisão (Precision)
    prec_0, prec_1, prec_avg = results['Prec(0)'], results['Prec(1)'], results['Prec(avg)'] # Linha 141
    print(f"Prec.   | {prec_0:.2f}% | {prec_1:.2f}% | {prec_avg:.2f}%") # Linha 142

    # Imprime Recall
    rec_0, rec_1, rec_avg = results['Rec(0)'], results['Rec(1)'], results['Rec(avg)'] # Linha 145
    print(f"Recall  | {rec_0:.2f}% | {rec_1:.2f}% | {rec_avg:.2f}%") # Linha 146

    # Imprime F1-Score
    f1_0, f1_1, f1_avg = results['F1(0)'], results['F1(1)'], results['F1(avg)'] # Linha 149
    print(f"F1-Score| {f1_0:.2f}% | {f1_1:.2f}% | {f1_avg:.2f}%") # Linha 150
    print("-" * 65) # Linha 151


# ==============================================================================
# 4. Execução Principal: OCSVM TREINADO sem SMOTE
# ==============================================================================

# 4.1 Carregar e Dividir Dados
X_raw, y, N_DAYS = load_sgcc_data(DATA_PATH) # Linha 157
# Separa 70% para treino e 30% para teste, mantendo a proporção de classes (stratify).
X_train_full, X_test, y_train_full, y_test = train_test_split(X_raw, y, test_size=0.3, random_state=SEED, stratify=y) # Linha 159
# Filtra apenas a classe normal (0) para treinamento One-Class
X_train_normal = X_train_full[y_train_full == 0] # Linha 161

print("\n" + "=" * 60) # Linha 163
print(f"Dados de Treino (Original): {X_train_full.shape}") # Linha 164
print(f"Distribuição de Classes no Treino: {Counter(y_train_full)}") # Linha 165
print(f"Amostras usadas para Treinamento OCSVM (Classe 0): {X_train_normal.shape[0]}") # Linha 166
print("=" * 60) # Linha 167

# 4.2 Pré-processamento
X_train_normal_scaled, scaler = preprocess_data(X_train_normal, fit_scaler=True) # Linha 170
X_test_scaled, _ = preprocess_data(X_test, fit_scaler=False, scaler=scaler) # Linha 171

# 4.3 Configuração e Treinamento do OCSVM
OCSVM_PARAMS = { # Linha 174
    'kernel': 'rbf',  # Tipo de kernel (função de similaridade)
    'nu': 0.1,        # Parâmetro de regularização (limite superior da fração de vetores de suporte/outliers).
    'gamma': 'auto',  # Parâmetro do kernel 'rbf'. 'auto' usa 1 / n_features.
} # Linha 178

print("\n--- OCSVM (None Sampling) - Configuração ---") # Linha 180
print("Parâmetros:", OCSVM_PARAMS) # Linha 181
print(f"Treinamento em {X_train_normal_scaled.shape[0]} amostras normais.") # Linha 182

model_ocsvm_none = OneClassSVM(**OCSVM_PARAMS) # Linha 184
model_ocsvm_none.fit(X_train_normal_scaled) # Linha 185

print("Treinamento do OCSVM (None Sampling) concluído.") # Linha 187

# 4.4 Avaliação
# O limiar (threshold) é ignorado no OCSVM, pois a predição é binária (+1 ou -1) baseada na distância da fronteira.
results_ocsvm_none = evaluate_anomaly_model(model_ocsvm_none, X_test_scaled, y_test) # Linha 191
print_results("OCSVM (None Sampling) - Resultados Detalhados no Teste", results_ocsvm_none) # Linha 192
print("=" * 60) # Linha 193
