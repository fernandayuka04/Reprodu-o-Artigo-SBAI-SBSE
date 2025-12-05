# ==============================================================================
# SCRIPT 8/8: Random Forest Classifier (RF) SEM SMOTE
# Objetivo: Treinar e avaliar o Random Forest nos dados de treino originais
#           (imbalanceados), usando o ajuste de peso interno ('class_weight').
# Dependência: sklearn
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. IMPORTAÇÕES NECESSÁRIAS
# ------------------------------------------------------------------------------
import pandas as pd  # Biblioteca para manipulação de DataFrames.
import numpy as np  # Biblioteca para operações numéricas.
from collections import Counter  # Para contagem e inspeção da distribuição das classes.
from sklearn.preprocessing import MinMaxScaler  # Normalização dos dados.
from sklearn.model_selection import train_test_split  # Para dividir os dados em treino e teste.
from sklearn.ensemble import RandomForestClassifier  # O modelo Random Forest.
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score  # Métricas de avaliação.
from scipy.interpolate import interp1d  # Para preenchimento de valores ausentes (interpolação).
import os  # Para manipulação de caminhos de arquivo.

# ------------------------------------------------------------------------------
# 2. CONFIGURAÇÕES INICIAIS
# ------------------------------------------------------------------------------
SEED = 42  # Semente para reprodutibilidade.
np.random.seed(SEED)  # Define a semente para operações do numpy.
os.environ['PYTHONHASHSEED'] = str(SEED)  # Define a semente para o hash Python.
# Simulação do caminho do arquivo de dados (ajuste se o caminho real for diferente)
DATA_PATH = os.path.join('.', 'data set.csv')


# ------------------------------------------------------------------------------
# 3. FUNÇÕES DE SUPORTE (CARREGAMENTO, PRÉ-PROCESSAMENTO, AVALIAÇÃO)
# ------------------------------------------------------------------------------

def load_sgcc_data(path):
    """Carrega o dataset real ou gera dados simulados se o arquivo não for encontrado."""
    try:
        # Tenta carregar o arquivo CSV real
        df = pd.read_csv(path)
        TARGET_COLUMN = df.columns[-1]
        y = df[TARGET_COLUMN].values
        X_data_raw = df.drop(columns=[TARGET_COLUMN])
        X_data = X_data_raw.apply(pd.to_numeric, errors='coerce').values
        N_DAYS = X_data.shape[1]
        print(f"Dados carregados: {X_data.shape[0]} consumidores, {N_DAYS} dias.")
        return X_data, y, N_DAYS
    except FileNotFoundError:
        # Caso o arquivo não exista, usa dados simulados
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
    # Imputação (preenche NaNs com interpolação linear ou 0)
    X_imputed = X_data.copy()
    for i in range(X_imputed.shape[0]):
        series = X_imputed[i, :]
        not_nan_indices = np.where(~np.isnan(series))[0]
        nan_indices = np.where(np.isnan(series))[0]
        if len(not_nan_indices) >= 2:
            interp_func = interp1d(not_nan_indices, series[not_nan_indices], kind='linear', fill_value='extrapolate')
            series[nan_indices] = interp_func(nan_indices)
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
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_outlier_handled)
        return X_scaled, scaler
    else:
        X_scaled = scaler.transform(X_outlier_handled)
        return X_scaled, scaler


def evaluate_classifier_model(model, X_test, y_test):
    """Avalia o modelo Classificador (Random Forest ou outro classificador binário)."""
    # O modelo já retorna as classes preditas (0 ou 1) diretamente com .predict()
    y_pred = model.predict(X_test)

    # Cálculo das métricas
    results = {
        'Acc(avg)': accuracy_score(y_test, y_pred) * 100,
        'Prec(avg)': precision_score(y_test, y_pred, average='macro', zero_division=0) * 100,
        'Rec(avg)': recall_score(y_test, y_pred, average='macro', zero_division=0) * 100,
        'F1(avg)': f1_score(y_test, y_pred, average='macro', zero_division=0) * 100,
        'Rec(1)': recall_score(y_test, y_pred, pos_label=1, zero_division=0) * 100,  # Recall Fraude (o mais importante)
    }
    return results


def print_results(title, results):
    """Imprime os resultados formatados."""
    print(f"\n--- {title} ---")
    print(pd.Series(results).apply(lambda x: f'{x:.2f}%'))
    print("-" * 50)


# ------------------------------------------------------------------------------
# 4. EXECUÇÃO PRINCIPAL: Random Forest Classifier TREINADO SEM SMOTE
# ------------------------------------------------------------------------------

# 4.1 Carregar e Dividir Dados
X_raw, y, N_DAYS = load_sgcc_data(DATA_PATH)
X_train_full, X_test, y_train_full, y_test = train_test_split(X_raw, y, test_size=0.3, random_state=SEED, stratify=y)

print("\n" + "=" * 60)
print(f"Dados de Treino (Original): {X_train_full.shape}")
initial_counts = Counter(y_train_full)
print(f"Distribuição de Classes no Treino: {initial_counts}")
print("=" * 60)

# 4.2 Pré-processamento e Normalização
X_train_scaled, scaler = preprocess_data(X_train_full, fit_scaler=True)
X_test_scaled, _ = preprocess_data(X_test, fit_scaler=False, scaler=scaler)

# 4.3 Ajuste de Classe para Imbalanceamento
print("\nModelo Random Forest será ajustado para dar mais peso à classe de Fraude (1) usando 'class_weight'.")
print("-" * 60)

# 4.4 Configuração e Treinamento do Random Forest Classifier
# Parâmetros otimizados para classificação em problemas de fraude.
RF_PARAMS = {
    'n_estimators': 150,  # [Parâmetro 1] Número de árvores na floresta (mais árvores = mais robustez).
    'max_depth': 15,  # [Parâmetro 2] Profundidade máxima das árvores (controla overfitting).
    'min_samples_split': 2,  # [Parâmetro 3] Número mínimo de amostras necessárias para dividir um nó interno.
    'min_samples_leaf': 1,  # [Parâmetro 4] Número mínimo de amostras necessárias em um nó folha.
    'criterion': 'gini',  # [Parâmetro 5] Função para medir a qualidade de uma divisão ('gini' ou 'entropy').
    # [Parâmetro de Ajuste de Imbalanceamento]
    'class_weight': 'balanced',  # [Parâmetro 6] Atribui pesos inversamente proporcionais à frequência das classes.
    'random_state': SEED,  # Semente para reprodutibilidade.
    'n_jobs': -1  # Usa todos os núcleos da CPU.
}

print("\n--- Random Forest (Sem SMOTE) - Configuração ---")
print("Parâmetros:", RF_PARAMS)
print(f"Treinamento em {X_train_scaled.shape[0]} amostras IMBALANCEADAS.")

# Treinamento: Utiliza o dataset de treino original e desbalanceado
model_rf_no_smote = RandomForestClassifier(**RF_PARAMS)
model_rf_no_smote.fit(X_train_scaled, y_train_full)

print("Treinamento do Random Forest (Sem SMOTE) concluído.")

# 4.5 Avaliação
results_rf_no_smote = evaluate_classifier_model(model_rf_no_smote, X_test_scaled, y_test)
print_results("Random Forest Classifier TREINADO SEM SMOTE - Resultados no Teste", results_rf_no_smote)
print("=" * 60)