# ==============================================================================
# SCRIPT 7/8: Random Forest Classifier (RF) COM SMOTE
# Objetivo: Treinar e avaliar o Random Forest, usando SMOTE para balancear as
#           classes de treino, em um cenário que simula o balanceamento de dados.
# Dependência: sklearn, imblearn
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. IMPORTAÇÕES NECESSÁRIAS
# ------------------------------------------------------------------------------
import pandas as pd  # Linha 11: Biblioteca para manipulação de DataFrames.
import numpy as np  # Linha 12: Biblioteca para operações numéricas.
from collections import Counter  # Linha 13: Para contagem e inspeção da distribuição das classes.
from imblearn.over_sampling import \
    SMOTE  # Linha 14: Técnica de Sobreamostragem de Minoria Sintética (para balanceamento).
from sklearn.preprocessing import MinMaxScaler  # Linha 15: Normalização dos dados.
from sklearn.model_selection import train_test_split  # Linha 16: Para dividir os dados em treino e teste.
from sklearn.ensemble import RandomForestClassifier  # Linha 17: O modelo Random Forest.
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score  # Linha 18: Métricas de avaliação.
from scipy.interpolate import interp1d  # Linha 19: Para preenchimento de valores ausentes (interpolação).
import os  # Linha 20: Para manipulação de caminhos de arquivo.

# ------------------------------------------------------------------------------
# 2. CONFIGURAÇÕES INICIAIS
# ------------------------------------------------------------------------------
SEED = 42  # Linha 24: Semente para reprodutibilidade.
np.random.seed(SEED)  # Linha 25: Define a semente para operações do numpy.
os.environ['PYTHONHASHSEED'] = str(SEED)  # Linha 26: Define a semente para o hash Python.
# Simulação do caminho do arquivo de dados (ajuste se o caminho real for diferente)
DATA_PATH = os.path.join('.', 'data set.csv')  # Linha 28


# ------------------------------------------------------------------------------
# 3. FUNÇÕES DE SUPORTE (CARREGAMENTO, PRÉ-PROCESSAMENTO, AVALIAÇÃO)
# ------------------------------------------------------------------------------

def load_sgcc_data(path):  # Linha 34
    """Carrega o dataset real ou gera dados simulados (com fraude sutil) se o arquivo não for encontrado."""
    try:
        # Tenta carregar o arquivo CSV real
        df = pd.read_csv(path)  # Linha 37
        TARGET_COLUMN = df.columns[-1]  # Linha 38
        y = df[TARGET_COLUMN].values  # Linha 39
        X_data_raw = df.drop(columns=[TARGET_COLUMN])  # Linha 40
        X_data = X_data_raw.apply(pd.to_numeric, errors='coerce').values  # Linha 41
        N_DAYS = X_data.shape[1]  # Linha 42
        print(f"Dados carregados: {X_data.shape[0]} consumidores, {N_DAYS} dias.")  # Linha 43
        return X_data, y, N_DAYS  # Linha 44
    except FileNotFoundError:
        # Caso o arquivo não exista, usa dados SIMULADOS com padrão de fraude MAIS COMPLEXO
        N_SAMPLES = 1000  # Linha 47: Número total de amostras (consumidores).
        N_DAYS_SIM = 1035  # Linha 48: Número de dias no histórico.
        FRAUD_RATE = 0.0853  # Linha 49: Taxa de fraude de 8.53%.
        time = np.linspace(0, 2 * np.pi, N_DAYS_SIM)  # Linha 50

        # Consumo Normal: Senoide + Ruído Significativo (STD=5)
        X_base = 50 + 20 * np.sin(time * 5)  # Linha 52
        X_normal = np.tile(X_base, (int(N_SAMPLES * (1 - FRAUD_RATE)), 1)) + np.random.normal(0, 5,
                                                                                              (int(N_SAMPLES * (
                                                                                                          1 - FRAUD_RATE)),
                                                                                               N_DAYS_SIM))  # Linha 53

        # Consumo Fraudulento (Simulação com padrões variáveis e sutis)
        X_fraud = np.tile(X_base, (int(N_SAMPLES * FRAUD_RATE), 1))  # Linha 57

        # APLICAÇÃO DE FRAUDE MAIS SUTIL E VARIÁVEL (para evitar 100% de acurácia)
        for i in range(X_fraud.shape[0]):  # Linha 60: Itera sobre as amostras fraudulentas.
            # A fraude começa aleatoriamente (após o dia 50)
            start_day = np.random.randint(50, N_DAYS_SIM - 200)  # Linha 62
            # A fraude dura entre 50 e 200 dias
            duration = np.random.randint(50, 200)  # Linha 64
            end_day = min(start_day + duration, N_DAYS_SIM)  # Linha 65

            # Fator de redução de consumo: menos drástico (entre 30% e 70% do consumo normal)
            reduction_factor = np.random.uniform(0.3, 0.7)  # Linha 68

            # Aplica o fator de redução na janela de fraude
            X_fraud[i, start_day:end_day] *= reduction_factor  # Linha 71

            # Adiciona ruído de tamanho similar ao X_normal para maior sobreposição
            X_fraud[i, :] += np.random.normal(0, 5, N_DAYS_SIM)  # Linha 74

        X_sim = np.vstack([X_normal, X_fraud])  # Linha 76: Combina dados normais e fraudulentos.
        y_sim = np.array([0] * X_normal.shape[0] + [1] * X_fraud.shape[
            0])  # Linha 77: Cria o vetor de rótulos (0: Normal, 1: Fraude).
        print("ARQUIVO NÃO ENCONTRADO. Usando dados SIMULADOS, AGORA MAIS COMPLEXOS.")  # Linha 78
        return X_sim, y_sim, N_DAYS_SIM  # Linha 79


def preprocess_data(X_data, fit_scaler=False, scaler=None):  # Linha 82: Função de pré-processamento.
    """Realiza interpolação linear, tratamento de outliers e normalização MinMaxScaler."""
    # Imputação (preenche NaNs com interpolação linear ou 0)
    X_imputed = X_data.copy()  # Linha 85: Cria uma cópia para não modificar o original.
    for i in range(X_imputed.shape[0]):  # Linha 86: Itera por cada série temporal (consumidor).
        series = X_imputed[i, :]  # Linha 87: Extrai a série atual.
        not_nan_indices = np.where(~np.isnan(series))[0]  # Linha 88: Índices dos valores válidos.
        nan_indices = np.where(np.isnan(series))[0]  # Linha 89: Índices dos valores ausentes (NaN).
        if len(not_nan_indices) >= 2:  # Linha 90: Interpolação só é possível com pelo menos 2 pontos.
            interp_func = interp1d(not_nan_indices, series[not_nan_indices], kind='linear',
                                   fill_value='extrapolate')  # Linha 91: Função de interpolação.
            series[nan_indices] = interp_func(nan_indices)  # Linha 92: Preenche os NaNs.
        series[np.isnan(series)] = 0  # Linha 93: Preenche NaNs restantes.
        X_imputed[i, :] = series  # Linha 94

    # Tratamento de Outliers (clipagem em Média + 2*Desvio Padrão)
    avg_x = np.mean(X_imputed, axis=0)  # Linha 97: Média de consumo para cada dia.
    std_x = np.std(X_imputed, axis=0)  # Linha 98: Desvio padrão para cada dia.
    threshold_outlier = avg_x + 2 * std_x  # Linha 99: Limite superior (Média + 2DP).
    X_outlier_handled = X_imputed.copy()  # Linha 100: Cópia para tratamento de outliers.

    for j in range(X_outlier_handled.shape[1]):  # Linha 102: Itera por cada dia (coluna).
        mask = X_outlier_handled[:, j] > threshold_outlier[j]  # Linha 103: Máscara para valores acima do limite.
        X_outlier_handled[mask, j] = threshold_outlier[j]  # Linha 104: Substitui outliers pelo limite (clipagem).

    # Normalização (MinMaxScaler)
    if fit_scaler:  # Linha 107: Se for para ajustar o scaler (apenas no treino).
        scaler = MinMaxScaler()  # Linha 108: Instancia o scaler.
        X_scaled = scaler.fit_transform(X_outlier_handled)  # Linha 109: Ajusta e transforma.
        return X_scaled, scaler  # Linha 110
    else:
        X_scaled = scaler.transform(X_outlier_handled)  # Linha 112: Apenas transforma (usado no teste).
        return X_scaled, scaler  # Linha 113


def evaluate_model(model, X_test, y_test):  # Linha 116: Função de avaliação detalhada.
    """
    Avalia o modelo Random Forest e calcula as métricas detalhadas (por classe e Média Macro).
    """
    y_pred = model.predict(X_test)  # Linha 122: Predição das classes no conjunto de teste.

    # 1. Cálculo das Métricas (usando sklearn.metrics)

    # Linha 126: Métrica: Acurácia geral
    accuracy = accuracy_score(y_test, y_pred) * 100

    # Linha 130-132: Métricas: Média Macro (average='macro')
    # Média não ponderada, essencial para dados desbalanceados.
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100

    # Linha 135-137: Métricas: Por Classe (average=None) -> [Valor Classe 0, Valor Classe 1]
    prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0) * 100
    rec_per_class = recall_score(y_test, y_pred, average=None, zero_division=0) * 100
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0) * 100

    results = {
        'Acc': accuracy,  # Linha 139
        'Prec(avg)': prec_macro,  # Linha 140
        'Rec(avg)': rec_macro,  # Linha 141
        'F1(avg)': f1_macro,  # Linha 142
        # Linha 144-149: Resultados por classe (Classe 0: Normal, Classe 1: Fraude)
        'Prec(0)': prec_per_class[0],
        'Prec(1)': prec_per_class[1],
        'Rec(0)': rec_per_class[0],
        'Rec(1)': rec_per_class[1],
        'F1(0)': f1_per_class[0],
        'F1(1)': f1_per_class[1],
    }
    return results  # Linha 150


def print_results(title, results):  # Linha 153: Função para imprimir resultados em formato de tabela.
    """Imprime os resultados detalhados das métricas no formato de tabela."""
    print(f"\n--- {title} ---")  # Linha 155
    print("Métrica | Classe 0 (Normal) | Classe 1 (Fraude) | Média Macro")  # Linha 156 (Cabeçalho da Tabela)
    print("-" * 65)  # Linha 157

    # Linha 160-162: Imprime Acurácia
    acc = results['Acc']
    print(f"Acc.    | {acc:.2f}% | {acc:.2f}% | {acc:.2f}%")

    # Linha 165-167: Imprime Precisão (Precision)
    prec_0, prec_1, prec_avg = results['Prec(0)'], results['Prec(1)'], results['Prec(avg)']
    print(f"Prec.   | {prec_0:.2f}% | {prec_1:.2f}% | {prec_avg:.2f}%")

    # Linha 170-172: Imprime Recall
    rec_0, rec_1, rec_avg = results['Rec(0)'], results['Rec(1)'], results['Rec(avg)']
    print(f"Recall  | {rec_0:.2f}% | {rec_1:.2f}% | {rec_avg:.2f}%")

    # Linha 175-177: Imprime F1-Score
    f1_0, f1_1, f1_avg = results['F1(0)'], results['F1(1)'], results['F1(avg)']
    print(f"F1-Score| {f1_0:.2f}% | {f1_1:.2f}% | {f1_avg:.2f}%")
    print("-" * 65)


# ------------------------------------------------------------------------------
# 4. EXECUÇÃO PRINCIPAL: Random Forest Classifier TREINADO com SMOTE
# ------------------------------------------------------------------------------

# 4.1 Carregar e Dividir Dados
X_raw, y, N_DAYS = load_sgcc_data(DATA_PATH)  # Linha 185: Carrega os dados (simulados ou reais).
X_train_full, X_test, y_train_full, y_test = train_test_split(X_raw, y, test_size=0.3, random_state=SEED,
                                                              stratify=y)  # Linha 186: Divide os dados mantendo a proporção de classes (stratify=y).

print("\n" + "=" * 60)  # Linha 188
print(f"Dados de Treino (Original): {X_train_full.shape}")  # Linha 189
print(f"Distribuição de Classes no Treino: {Counter(y_train_full)}")  # Linha 190
print("=" * 60)  # Linha 191

# 4.2 Pré-processamento e Normalização
X_train_scaled, scaler = preprocess_data(X_train_full,
                                         fit_scaler=True)  # Linha 194: Ajusta o scaler e transforma o treino.
X_test_scaled, _ = preprocess_data(X_test, fit_scaler=False,
                                   scaler=scaler)  # Linha 195: Transforma o teste usando o scaler ajustado.

# 4.3 Aplicação do SMOTE (Apenas nos dados de TREINO)
print("Aplicando SMOTE aos dados de Treinamento...")  # Linha 198
sm = SMOTE(random_state=SEED)  # Linha 199: Instancia o SMOTE.
X_res, y_res = sm.fit_resample(X_train_scaled, y_train_full)  # Linha 200: Gera amostras sintéticas (balanceamento).

print("-" * 60)  # Linha 202
print("Distribuição de Classes no Treino DEPOIS do SMOTE:", Counter(y_res))  # Linha 203: Exibe o balanceamento.
print(f"Número total de amostras no treino depois do SMOTE: {len(X_res)}")  # Linha 204
print("-" * 60)  # Linha 205

# 4.4 Configuração e Treinamento do Random Forest Classifier
# O parâmetro 'class_weight' NÃO é usado, pois as classes já foram balanceadas pelo SMOTE.
# [Parâmetros OTIMIZADOS inseridos abaixo]
RF_PARAMS = {  # Linha 210: Dicionário de hiperparâmetros.
    'n_estimators': 200,  # Linha 211: [ALTERADO/Otimizado] Número de árvores na floresta (200).
    'max_depth': 20,  # Linha 212: [ALTERADO/Otimizado] Profundidade máxima das árvores (20).
    'min_samples_split': 2,  # Linha 213: Mínimo de amostras para dividir um nó interno.
    'min_samples_leaf': 1,  # Linha 214: Mínimo de amostras em um nó folha.
    'criterion': 'gini',  # Linha 215: Função para medir a qualidade de uma divisão.
    # Note: 'class_weight' é omitido porque o SMOTE já balanceou os dados de treino.
    'random_state': SEED,  # Linha 217: Semente.
    'n_jobs': -1  # Linha 218: Usa todos os núcleos da CPU.
}

print("\n--- Random Forest (SMOTE) - Configuração ---")  # Linha 220
print("Parâmetros:", RF_PARAMS)  # Linha 221: Exibe os parâmetros otimizados.
print(f"Treinamento em {X_res.shape[0]} amostras BALANCEADAS.")  # Linha 222

# Treinamento: Utiliza o dataset BALANCEADO (X_res, y_res)
model_rf_smote = RandomForestClassifier(**RF_PARAMS)  # Linha 225: Instancia o modelo com os parâmetros.
model_rf_smote.fit(X_res, y_res)  # Linha 226: Treina o modelo nos dados reamostrados.

print("Treinamento do Random Forest (SMOTE) concluído.")  # Linha 228

# 4.5 Avaliação
# A avaliação é feita no conjunto de teste original (X_test_scaled), não reamostrado
results_rf_smote = evaluate_model(model_rf_smote, X_test_scaled, y_test)  # Linha 232
print_results("Random Forest Classifier TREINADO com SMOTE - Resultados Detalhados no Teste",
              results_rf_smote)  # Linha 233
print("=" * 60)  # Linha 234
