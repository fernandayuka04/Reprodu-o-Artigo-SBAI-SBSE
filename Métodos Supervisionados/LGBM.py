# ==============================================================================
# SCRIPT 6/6: Light Gradient Boosting Machine (LGBM) SEM SMOTE
# Objetivo: Treinar e avaliar o LGBM como um classificador binário, usando os
#           dados de treino originais (imbalanceados), ajustando o peso da
#           classe minoritária para lidar com o desequilíbrio.
# Dependência: lightgbm, scikit-learn
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. IMPORTAÇÕES NECESSÁRIAS
# ------------------------------------------------------------------------------
import pandas as pd  # Linha 11: Biblioteca para manipulação de DataFrames.
import numpy as np  # Linha 12: Biblioteca para operações numéricas.
from collections import Counter  # Linha 13: Para contagem e inspeção da distribuição das classes.
from sklearn.preprocessing import MinMaxScaler  # Linha 14: Normalização dos dados.
from sklearn.model_selection import train_test_split  # Linha 15: Para dividir os dados em treino e teste.
from lightgbm import LGBMClassifier  # Linha 16: O modelo LightGBM.
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score  # Linha 17: Métricas de avaliação.
from scipy.interpolate import interp1d  # Linha 18: Para preenchimento de valores ausentes (interpolação).
import os  # Linha 19: Para manipulação de caminhos de arquivo.

# ------------------------------------------------------------------------------
# 2. CONFIGURAÇÕES INICIAIS
# ------------------------------------------------------------------------------
SEED = 42  # Linha 23: Semente para reprodutibilidade.
np.random.seed(SEED)  # Linha 24: Define a semente para operações do numpy.
os.environ['PYTHONHASHSEED'] = str(SEED)  # Linha 25: Define a semente para o hash Python.
# Simulação do caminho do arquivo de dados (ajuste se o caminho real for diferente)
DATA_PATH = os.path.join('.', 'data set.csv')  # Linha 27


# ------------------------------------------------------------------------------
# 3. FUNÇÕES DE SUPORTE (CARREGAMENTO, PRÉ-PROCESSAMENTO, AVALIAÇÃO)
# ------------------------------------------------------------------------------

def load_sgcc_data(path):  # Linha 33
    """Carrega o dataset real ou gera dados simulados se o arquivo não for encontrado."""
    try:
        # Tenta carregar o arquivo CSV real
        df = pd.read_csv(path)  # Linha 36
        TARGET_COLUMN = df.columns[-1]  # Linha 37
        y = df[TARGET_COLUMN].values  # Linha 38
        X_data_raw = df.drop(columns=[TARGET_COLUMN])  # Linha 39
        X_data = X_data_raw.apply(pd.to_numeric, errors='coerce').values  # Linha 40
        N_DAYS = X_data.shape[1]  # Linha 41
        print(f"Dados carregados: {X_data.shape[0]} consumidores, {N_DAYS} dias.")  # Linha 42
        return X_data, y, N_DAYS  # Linha 43
    except FileNotFoundError:
        # Caso o arquivo não exista, usa dados simulados
        N_SAMPLES = 1000;
        N_DAYS_SIM = 1035;
        FRAUD_RATE = 0.0853  # Linha 46
        time = np.linspace(0, 2 * np.pi, N_DAYS_SIM)  # Linha 47
        X_normal = np.tile(50 + 20 * np.sin(time * 5), (int(N_SAMPLES * (1 - FRAUD_RATE)), 1)) + np.random.normal(0, 5,
                                                                                                                  (int(
                                                                                                                      N_SAMPLES * (
                                                                                                                                  1 - FRAUD_RATE)),
                                                                                                                   N_DAYS_SIM))  # Linha 48
        X_fraud = np.tile(50 + 20 * np.sin(time * 5), (int(N_SAMPLES * FRAUD_RATE), 1))  # Linha 52
        for i in range(X_fraud.shape[0]):  # Linha 53
            X_fraud[i, :np.random.randint(200, N_DAYS_SIM)] *= np.random.uniform(0.1, 0.5)  # Linha 54
            X_fraud[i, :] += np.random.normal(0, 2, N_DAYS_SIM)  # Linha 55
        X_sim = np.vstack([X_normal, X_fraud]);
        y_sim = np.array([0] * X_normal.shape[0] + [1] * X_fraud.shape[0])  # Linha 56
        print("ARQUIVO NÃO ENCONTRADO. Usando dados SIMULADOS.")  # Linha 57
        return X_sim, y_sim, N_DAYS_SIM  # Linha 58


def preprocess_data(X_data, fit_scaler=False, scaler=None):  # Linha 61
    """Realiza interpolação linear, tratamento de outliers e normalização MinMaxScaler."""
    # Imputação (preenche NaNs com interpolação linear ou 0)
    X_imputed = X_data.copy()  # Linha 64
    for i in range(X_imputed.shape[0]):  # Linha 65
        series = X_imputed[i, :]  # Linha 66
        not_nan_indices = np.where(~np.isnan(series))[0]  # Linha 67
        nan_indices = np.where(np.isnan(series))[0]  # Linha 68
        if len(not_nan_indices) >= 2:  # Linha 69
            interp_func = interp1d(not_nan_indices, series[not_nan_indices], kind='linear', fill_value='extrapolate')  # Linha 70
            series[nan_indices] = interp_func(nan_indices)  # Linha 71
        series[np.isnan(series)] = 0  # Linha 72
        X_imputed[i, :] = series  # Linha 73

    # Tratamento de Outliers (clipagem em Média + 2*Desvio Padrão)
    avg_x = np.mean(X_imputed, axis=0)  # Linha 76
    std_x = np.std(X_imputed, axis=0)  # Linha 77
    threshold_outlier = avg_x + 2 * std_x  # Linha 78
    X_outlier_handled = X_imputed.copy()  # Linha 79

    for j in range(X_outlier_handled.shape[1]):  # Linha 81
        mask = X_outlier_handled[:, j] > threshold_outlier[j]  # Linha 82
        X_outlier_handled[mask, j] = threshold_outlier[j]  # Linha 83

    # Normalização (MinMaxScaler)
    if fit_scaler:  # Linha 86
        scaler = MinMaxScaler()  # Linha 87
        X_scaled = scaler.fit_transform(X_outlier_handled)  # Linha 88
        return X_scaled, scaler  # Linha 89
    else:
        X_scaled = scaler.transform(X_outlier_handled)  # Linha 91
        return X_scaled, scaler  # Linha 92


def evaluate_model(model, X_test, y_test):  # Linha 95: Função de avaliação detalhada para o LGBM.
    """
    Avalia o modelo LGBM e calcula as métricas detalhadas (por classe e Média Macro).
    O LGBM retorna as classes preditas (0 ou 1) com .predict().
    """
    y_pred = model.predict(X_test)  # Linha 101

    # 1. Cálculo das Métricas (usando sklearn.metrics)

    # Métrica: Acurácia geral
    accuracy = accuracy_score(y_test, y_pred) * 100  # Linha 106

    # Métricas: Média Macro (average='macro')
    # A média macro trata igualmente ambas as classes, ideal para dados desbalanceados.
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100  # Linha 110
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100  # Linha 111
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100  # Linha 112

    # Métricas: Por Classe (average=None) -> [Valor Classe 0, Valor Classe 1]
    prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0) * 100  # Linha 115
    rec_per_class = recall_score(y_test, y_pred, average=None, zero_division=0) * 100  # Linha 116
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0) * 100  # Linha 117

    results = {
        'Acc': accuracy,  # Linha 119
        'Prec(avg)': prec_macro,  # Linha 120
        'Rec(avg)': rec_macro,  # Linha 121
        'F1(avg)': f1_macro,  # Linha 122
        # Resultados por classe
        'Prec(0)': prec_per_class[0],  # Linha 124 (Classe 0: Normal)
        'Prec(1)': prec_per_class[1],  # Linha 125 (Classe 1: Fraude)
        'Rec(0)': rec_per_class[0],  # Linha 126
        'Rec(1)': rec_per_class[1],  # Linha 127
        'F1(0)': f1_per_class[0],  # Linha 128
        'F1(1)': f1_per_class[1],  # Linha 129
    }
    return results  # Linha 130


def print_results(title, results):  # Linha 133
    """Imprime os resultados detalhados das métricas no formato de tabela."""
    print(f"\n--- {title} ---")  # Linha 135
    print("Métrica | Classe 0 (Normal) | Classe 1 (Fraude) | Média Macro")  # Linha 136 (Cabeçalho da Tabela)
    print("-" * 65)  # Linha 137

    # Imprime Acurácia (é um valor único)
    acc = results['Acc']  # Linha 140
    print(f"Acc.    | {acc:.2f}% | {acc:.2f}% | {acc:.2f}%")  # Linha 141

    # Imprime Precisão (Precision)
    prec_0, prec_1, prec_avg = results['Prec(0)'], results['Prec(1)'], results['Prec(avg)']  # Linha 144
    print(f"Prec.   | {prec_0:.2f}% | {prec_1:.2f}% | {prec_avg:.2f}%")  # Linha 145

    # Imprime Recall
    rec_0, rec_1, rec_avg = results['Rec(0)'], results['Rec(1)'], results['Rec(avg)']  # Linha 148
    print(f"Recall  | {rec_0:.2f}% | {rec_1:.2f}% | {rec_avg:.2f}%")  # Linha 149

    # Imprime F1-Score
    f1_0, f1_1, f1_avg = results['F1(0)'], results['F1(1)'], results['F1(avg)']  # Linha 152
    print(f"F1-Score| {f1_0:.2f}% | {f1_1:.2f}% | {f1_avg:.2f}%")  # Linha 153
    print("-" * 65)  # Linha 154


# ------------------------------------------------------------------------------
# 4. EXECUÇÃO PRINCIPAL: LGBM Classifier TREINADO SEM SMOTE
# ------------------------------------------------------------------------------

# 4.1 Carregar e Dividir Dados
# Carrega os dados (reais ou simulados).
X_raw, y, N_DAYS = load_sgcc_data(DATA_PATH)  # Linha 160
# Divide os dados. Stratify garante que a proporção de classes seja mantida.
X_train_full, X_test, y_train_full, y_test = train_test_split(X_raw, y, test_size=0.3, random_state=SEED, stratify=y)  # Linha 162

print("\n" + "=" * 60)  # Linha 164
print(f"Dados de Treino (Original): {X_train_full.shape}")  # Linha 165
initial_counts = Counter(y_train_full)  # Linha 166
print(f"Distribuição de Classes no Treino: {initial_counts}")  # Linha 167
print("=" * 60)  # Linha 168

# 4.2 Pré-processamento e Normalização
# Ajusta o scaler SOMENTE nos dados de treino.
X_train_scaled, scaler = preprocess_data(X_train_full, fit_scaler=True)  # Linha 171
# Transforma os dados de teste usando o scaler ajustado.
X_test_scaled, _ = preprocess_data(X_test, fit_scaler=False, scaler=scaler)  # Linha 173

# 4.3 Ajuste para o Desequilíbrio de Classes
# Linha 177: O parâmetro 'class_weight' do LGBM fará o ajuste de peso
# Linha 178: da classe minoritária (Fraude=1) automaticamente.
# Linha 179: O cálculo manual da razão de desequilíbrio (IMBALANCE_RATIO) foi removido.
# Linha 180
# Linha 181
# Linha 182
print("\nO modelo LGBM utilizará 'class_weight: balanced' para ajustar o peso da classe de Fraude (1).") # Linha 184
print("-" * 60)  # Linha 185
# Linha 186

# 4.4 Configuração e Treinamento do LGBM Classifier
# Parâmetros otimizados, conforme solicitado (GridSearchCV 3-fold cross-validation).
LGBM_PARAMS = {  # Linha 189: Dicionário com os hiperparâmetros OTIMIZADOS.
    'objective': 'binary',  # Linha 190: Define o problema como classificação binária.
    'metric': 'binary_logloss',  # Linha 191: Métrica para otimização (perda logarítmica binária).
    'boosting_type': 'gbdt',  # Linha 192: Tipo de boosting.
    'n_estimators': 300,  # Linha 193: Número de árvores/iterações (Atualizado: 300).
    'learning_rate': 0.2,  # Linha 194: Taxa de aprendizado (Atualizado: 0.2).
    'num_leaves': 70,  # Linha 195: Número máximo de folhas por árvore (Atualizado: 70).
    'max_depth': 30,  # Linha 196: Profundidade máxima da árvore (Atualizado: 30).
    'min_child_samples': 30,  # Linha 197: Mínimo de amostras em um nó filho (Atualizado: 30).
    'colsample_bytree': 0.8,  # Linha 198: Proporção de colunas amostradas.
    'subsample': 0.8,  # Linha 199: Proporção de dados amostrados.
    # [Parâmetro de Ajuste de Imbalanceamento]
    'class_weight': 'balanced',  # Linha 201: Usa pesos inversamente proporcionais às frequências das classes (conforme solicitado).
    'random_state': SEED,  # Linha 202: Semente para reprodutibilidade.
    'n_jobs': -1  # Linha 203: Usa todos os núcleos da CPU.
}

print("\n--- LGBM (Sem SMOTE) - Configuração ---")  # Linha 205
print("Parâmetros:", LGBM_PARAMS)  # Linha 206
print(f"Treinamento em {X_train_scaled.shape[0]} amostras IMBALANCEADAS.")  # Linha 207

# Treinamento: Utiliza o dataset de treino original e desbalanceado
model_lgbm_no_smote = LGBMClassifier(**LGBM_PARAMS)  # Linha 210
model_lgbm_no_smote.fit(X_train_scaled, y_train_full)  # Linha 211

print("Treinamento do LGBM (Sem SMOTE) concluído.")  # Linha 213

# 4.5 Avaliação
# A avaliação é feita no conjunto de teste original
results_lgbm_no_smote = evaluate_model(model_lgbm_no_smote, X_test_scaled, y_test)  # Linha 217
print_results("LGBM Classifier TREINADO SEM SMOTE - Resultados Detalhados no Teste", results_lgbm_no_smote)  # Linha 218
print("=" * 60)  # Linha 219
