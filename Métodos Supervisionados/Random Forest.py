# ==============================================================================
# SCRIPT 8/8: Random Forest Classifier (RF) SEM SMOTE
# Objetivo: Treinar e avaliar o Random Forest nos dados de treino originais
#           (imbalanceados), usando o ajuste de peso interno ('class_weight').
# Dependência: sklearn
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. IMPORTAÇÕES NECESSÁRIAS
# ------------------------------------------------------------------------------
import pandas as pd  # Linha 11: Biblioteca para manipulação de DataFrames.
import numpy as np  # Linha 12: Biblioteca para operações numéricas.
from collections import Counter  # Linha 13: Para contagem e inspeção da distribuição das classes.
from sklearn.preprocessing import MinMaxScaler  # Linha 14: Normalização dos dados (MinMaxScaler).
from sklearn.model_selection import train_test_split  # Linha 15: Para dividir os dados em treino e teste.
from sklearn.ensemble import RandomForestClassifier  # Linha 16: O modelo Random Forest.
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score  # Linha 17: Métricas de avaliação.
from scipy.interpolate import interp1d  # Linha 18: Para preenchimento de valores ausentes (interpolação).
import os  # Linha 19: Para manipulação de caminhos de arquivo.

# ------------------------------------------------------------------------------
# 2. CONFIGURAÇÕES INICIAIS
# ------------------------------------------------------------------------------
SEED = 42  # Linha 23: Semente para reprodutibilidade dos resultados.
np.random.seed(SEED)  # Linha 24: Define a semente para operações do numpy.
os.environ['PYTHONHASHSEED'] = str(SEED)  # Linha 25: Define a semente para o hash Python.
# Simulação do caminho do arquivo de dados (ajuste se o caminho real for diferente)
DATA_PATH = os.path.join('.', 'data set.csv')  # Linha 27


# ------------------------------------------------------------------------------
# 3. FUNÇÕES DE SUPORTE (CARREGAMENTO, PRÉ-PROCESSAMENTO, AVALIAÇÃO)
# ------------------------------------------------------------------------------

def load_sgcc_data(path):  # Linha 33
    """Carrega o dataset real ou gera dados simulados (com fraude sutil) se o arquivo não for encontrado."""
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
        # Caso o arquivo não exista, usa dados SIMULADOS com padrão de fraude MAIS COMPLEXO
        N_SAMPLES = 1000  # Linha 46: Número total de amostras (consumidores).
        N_DAYS_SIM = 1035  # Linha 47: Número de dias no histórico.
        FRAUD_RATE = 0.0853  # Linha 48: Taxa de fraude de 8.53%.
        time = np.linspace(0, 2 * np.pi, N_DAYS_SIM)  # Linha 49

        # Consumo Normal: Senoide + Ruído Significativo (STD=5)
        X_base = 50 + 20 * np.sin(time * 5)
        X_normal = np.tile(X_base, (int(N_SAMPLES * (1 - FRAUD_RATE)), 1)) + np.random.normal(0, 5,
                                                                                              (int(N_SAMPLES * (
                                                                                                          1 - FRAUD_RATE)),
                                                                                               N_DAYS_SIM))

        # Consumo Fraudulento (Simulação com padrões variáveis e sutis)
        X_fraud = np.tile(X_base, (int(N_SAMPLES * FRAUD_RATE), 1))

        # APLICAÇÃO DE FRAUDE MAIS SUTIL E VARIÁVEL (para evitar 100% de acurácia)
        for i in range(X_fraud.shape[0]):  # Linha 57
            # A fraude começa aleatoriamente (após o dia 50)
            start_day = np.random.randint(50, N_DAYS_SIM - 200)
            # A fraude dura entre 50 e 200 dias
            duration = np.random.randint(50, 200)
            end_day = min(start_day + duration, N_DAYS_SIM)

            # Fator de redução de consumo: menos drástico (entre 30% e 70% do consumo normal)
            reduction_factor = np.random.uniform(0.3, 0.7)

            # Aplica o fator de redução na janela de fraude
            X_fraud[i, start_day:end_day] *= reduction_factor

            # Adiciona ruído de tamanho similar ao X_normal para maior sobreposição
            X_fraud[i, :] += np.random.normal(0, 5, N_DAYS_SIM)

        X_sim = np.vstack([X_normal, X_fraud])  # Linha 68: Combina dados normais e fraudulentos.
        y_sim = np.array([0] * X_normal.shape[0] + [1] * X_fraud.shape[
            0])  # Linha 69: Cria o vetor de rótulos (0: Normal, 1: Fraude).
        print("ARQUIVO NÃO ENCONTRADO. Usando dados SIMULADOS, AGORA MAIS COMPLEXOS.")  # Linha 70
        return X_sim, y_sim, N_DAYS_SIM  # Linha 71


def preprocess_data(X_data, fit_scaler=False, scaler=None):  # Linha 74: Função de pré-processamento.
    """Realiza interpolação linear, tratamento de outliers e normalização MinMaxScaler."""
    # Imputação (preenche NaNs com interpolação linear ou 0)
    X_imputed = X_data.copy()  # Linha 77: Cria uma cópia para não modificar o original.
    for i in range(X_imputed.shape[0]):  # Linha 78: Itera por cada série temporal (consumidor).
        series = X_imputed[i, :]  # Linha 79: Extrai a série atual.
        not_nan_indices = np.where(~np.isnan(series))[0]  # Linha 80: Índices dos valores válidos.
        nan_indices = np.where(np.isnan(series))[0]  # Linha 81: Índices dos valores ausentes (NaN).
        if len(not_nan_indices) >= 2:  # Linha 82: Interpolação só é possível com pelo menos 2 pontos.
            interp_func = interp1d(not_nan_indices, series[not_nan_indices], kind='linear',
                                   fill_value='extrapolate')  # Linha 83: Função de interpolação.
            series[nan_indices] = interp_func(nan_indices)  # Linha 84: Preenche os NaNs.
        series[np.isnan(series)] = 0  # Linha 85: Preenche NaNs restantes (se houver).
        X_imputed[i, :] = series  # Linha 86

    # Tratamento de Outliers (clipagem em Média + 2*Desvio Padrão)
    avg_x = np.mean(X_imputed, axis=0)  # Linha 89: Média de consumo para cada dia.
    std_x = np.std(X_imputed, axis=0)  # Linha 90: Desvio padrão para cada dia.
    threshold_outlier = avg_x + 2 * std_x  # Linha 91: Limite superior (Média + 2DP).
    X_outlier_handled = X_imputed.copy()  # Linha 92: Cópia para tratamento de outliers.

    for j in range(X_outlier_handled.shape[1]):  # Linha 94: Itera por cada dia (coluna).
        mask = X_outlier_handled[:, j] > threshold_outlier[j]  # Linha 95: Máscara para valores acima do limite.
        X_outlier_handled[mask, j] = threshold_outlier[j]  # Linha 96: Substitui outliers pelo limite (clipagem).

    # Normalização (MinMaxScaler)
    if fit_scaler:  # Linha 99: Se for para ajustar o scaler (apenas no treino).
        scaler = MinMaxScaler()  # Linha 100: Instancia o scaler.
        X_scaled = scaler.fit_transform(X_outlier_handled)  # Linha 101: Ajusta e transforma.
        return X_scaled, scaler  # Linha 102
    else:
        X_scaled = scaler.transform(X_outlier_handled)  # Linha 104: Apenas transforma (usado no teste).
        return X_scaled, scaler  # Linha 105


def evaluate_model(model, X_test, y_test):  # Linha 108: Função de avaliação detalhada.
    """
    Avalia o modelo Random Forest e calcula as métricas detalhadas (por classe e Média Macro).
    """
    y_pred = model.predict(X_test)  # Linha 114: Predição das classes no conjunto de teste.

    # 1. Cálculo das Métricas (usando sklearn.metrics)

    # Linha 118: Métrica: Acurácia geral
    accuracy = accuracy_score(y_test, y_pred) * 100

    # Linha 122-124: Métricas: Média Macro (average='macro')
    # Média não ponderada, essencial para dados desbalanceados.
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100

    # Linha 127-129: Métricas: Por Classe (average=None) -> [Valor Classe 0, Valor Classe 1]
    prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0) * 100
    rec_per_class = recall_score(y_test, y_pred, average=None, zero_division=0) * 100
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0) * 100

    results = {
        'Acc': accuracy,  # Linha 131
        'Prec(avg)': prec_macro,  # Linha 132
        'Rec(avg)': rec_macro,  # Linha 133
        'F1(avg)': f1_macro,  # Linha 134
        # Linha 136-141: Resultados por classe (Classe 0: Normal, Classe 1: Fraude)
        'Prec(0)': prec_per_class[0],
        'Prec(1)': prec_per_class[1],
        'Rec(0)': rec_per_class[0],
        'Rec(1)': rec_per_class[1],
        'F1(0)': f1_per_class[0],
        'F1(1)': f1_per_class[1],
    }
    return results  # Linha 142


def print_results(title, results):  # Linha 145: Função para imprimir resultados em formato de tabela.
    """Imprime os resultados detalhados das métricas no formato de tabela."""
    print(f"\n--- {title} ---")  # Linha 147
    print("Métrica | Classe 0 (Normal) | Classe 1 (Fraude) | Média Macro")  # Linha 148 (Cabeçalho da Tabela)
    print("-" * 65)  # Linha 149

    # Linha 152-154: Imprime Acurácia
    acc = results['Acc']
    print(f"Acc.    | {acc:.2f}% | {acc:.2f}% | {acc:.2f}%")

    # Linha 157-159: Imprime Precisão (Precision)
    prec_0, prec_1, prec_avg = results['Prec(0)'], results['Prec(1)'], results['Prec(avg)']
    print(f"Prec.   | {prec_0:.2f}% | {prec_1:.2f}% | {prec_avg:.2f}%")

    # Linha 162-164: Imprime Recall
    rec_0, rec_1, rec_avg = results['Rec(0)'], results['Rec(1)'], results['Rec(avg)']
    print(f"Recall  | {rec_0:.2f}% | {rec_1:.2f}% | {rec_avg:.2f}%")

    # Linha 167-169: Imprime F1-Score
    f1_0, f1_1, f1_avg = results['F1(0)'], results['F1(1)'], results['F1(avg)']
    print(f"F1-Score| {f1_0:.2f}% | {f1_1:.2f}% | {f1_avg:.2f}%")
    print("-" * 65)


# ------------------------------------------------------------------------------
# 4. EXECUÇÃO PRINCIPAL: Random Forest Classifier TREINADO SEM SMOTE
# ------------------------------------------------------------------------------

# 4.1 Carregar e Dividir Dados
X_raw, y, N_DAYS = load_sgcc_data(DATA_PATH)  # Linha 177: Carrega os dados (simulados ou reais).
X_train_full, X_test, y_train_full, y_test = train_test_split(X_raw, y, test_size=0.3, random_state=SEED,
                                                              stratify=y)  # Linha 178: Divide os dados mantendo a proporção de classes (stratify=y).

print("\n" + "=" * 60)  # Linha 180
print(f"Dados de Treino (Original): {X_train_full.shape}")  # Linha 181
initial_counts = Counter(y_train_full)  # Linha 182
print(f"Distribuição de Classes no Treino: {initial_counts}")  # Linha 183
print("=" * 60)  # Linha 184

# 4.2 Pré-processamento e Normalização
X_train_scaled, scaler = preprocess_data(X_train_full, fit_scaler=True)  # Linha 187: Ajusta e transforma treino.
X_test_scaled, _ = preprocess_data(X_test, fit_scaler=False, scaler=scaler)  # Linha 188: Transforma teste.

# 4.3 Ajuste de Classe para Imbalanceamento
print(
    "\nModelo Random Forest será ajustado para dar mais peso à classe de Fraude (1) usando 'class_weight'.")  # Linha 191
print("-" * 60)  # Linha 192

# 4.4 Configuração e Treinamento do Random Forest Classifier
# Parâmetros OTIMIZADOS.
RF_PARAMS = {  # Linha 196: Dicionário de hiperparâmetros.
    # [ALTERADO] Número de árvores no Random Forest (antes: 150)
    'n_estimators': 200,  # Linha 198: Otimizado: 200 árvores.
    # [ALTERADO] Profundidade máxima das árvores (antes: 15)
    'max_depth': 20,  # Linha 200: Otimizado: Profundidade 20.
    'min_samples_split': 2,  # Linha 201: Mínimo de amostras para dividir um nó interno.
    'min_samples_leaf': 1,  # Linha 202: Mínimo de amostras em um nó folha.
    'criterion': 'gini',  # Linha 203: Função para medir a qualidade de uma divisão.
    # [Parâmetro de Ajuste de Imbalanceamento]
    'class_weight': 'balanced',  # Linha 205: Atribui pesos inversamente proporcionais à frequência das classes.
    'random_state': SEED,  # Linha 206: Semente.
    'n_jobs': -1  # Linha 207: Usa todos os núcleos da CPU.
}

print("\n--- Random Forest (Sem SMOTE) - Configuração ---")  # Linha 209
print("Parâmetros:", RF_PARAMS)  # Linha 210: Exibe os parâmetros otimizados.
print(f"Treinamento em {X_train_scaled.shape[0]} amostras IMBALANCEADAS.")  # Linha 211

# Treinamento: Utiliza o dataset de treino original e desbalanceado (o balanceamento é feito via 'class_weight')
model_rf_no_smote = RandomForestClassifier(**RF_PARAMS)  # Linha 214: Instancia o modelo com os parâmetros.
model_rf_no_smote.fit(X_train_scaled, y_train_full)  # Linha 215: Treina o modelo.

print("Treinamento do Random Forest (Sem SMOTE) concluído.")  # Linha 217

# 4.5 Avaliação
# A avaliação é feita no conjunto de teste original
results_rf_no_smote = evaluate_model(model_rf_no_smote, X_test_scaled, y_test)  # Linha 221
print_results("Random Forest Classifier TREINADO SEM SMOTE (Class Weight) - Resultados Detalhados no Teste",
              results_rf_no_smote)  # Linha 222
print("=" * 60)  # Linha 223
