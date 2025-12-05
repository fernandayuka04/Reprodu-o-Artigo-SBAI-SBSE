# ==============================================================================
# SCRIPT 5/6: Light Gradient Boosting Machine (LGBM) COM SMOTE
# Objetivo: Treinar e avaliar o LGBM como um classificador binário, usando SMOTE
#           para balancear as classes de treino antes da aplicação do modelo.
# Dependência: lightgbm, imblearn, scikit-learn
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
from lightgbm import LGBMClassifier  # Linha 17: O modelo LightGBM.
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
    """Carrega o dataset real ou gera dados simulados se o arquivo não for encontrado."""
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
        # Caso o arquivo não exista, usa dados SIMULADOS com um padrão de fraude mais complexo
        N_SAMPLES = 1000;
        N_DAYS_SIM = 1035;
        FRAUD_RATE = 0.0853  # Linha 47
        time = np.linspace(0, 2 * np.pi, N_DAYS_SIM)  # Linha 48

        # Consumo Normal: Senoide + Ruído Significativo (STD=5)
        X_base = 50 + 20 * np.sin(time * 5)
        X_normal = np.tile(X_base, (int(N_SAMPLES * (1 - FRAUD_RATE)), 1)) + np.random.normal(0, 5,
                                                                                              (int(N_SAMPLES * (
                                                                                                          1 - FRAUD_RATE)),
                                                                                               N_DAYS_SIM))

        # Consumo Fraudulento (Inicial): Senoide
        X_fraud = np.tile(X_base, (int(N_SAMPLES * FRAUD_RATE), 1))

        # APLICAÇÃO DE FRAUDE MAIS SUTIL E VARIÁVEL (para evitar 100% de acurácia)
        for i in range(X_fraud.shape[0]):  # Linha 54
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

        X_sim = np.vstack([X_normal, X_fraud]);
        y_sim = np.array([0] * X_normal.shape[0] + [1] * X_fraud.shape[0])  # Linha 57
        print("ARQUIVO NÃO ENCONTRADO. Usando dados SIMULADOS, AGORA MAIS COMPLEXOS.")  # Linha 58
        return X_sim, y_sim, N_DAYS_SIM  # Linha 59


def preprocess_data(X_data, fit_scaler=False, scaler=None):  # Linha 62
    """Realiza interpolação linear, tratamento de outliers e normalização MinMaxScaler."""
    # Imputação (preenche NaNs com interpolação linear ou 0)
    X_imputed = X_data.copy()  # Linha 65
    for i in range(X_imputed.shape[0]):  # Linha 66
        series = X_imputed[i, :]  # Linha 67
        not_nan_indices = np.where(~np.isnan(series))[0]  # Linha 68
        nan_indices = np.where(np.isnan(series))[0]  # Linha 69
        if len(not_nan_indices) >= 2:  # Linha 70
            interp_func = interp1d(not_nan_indices, series[not_nan_indices], kind='linear',
                                   fill_value='extrapolate')  # Linha 71
            series[nan_indices] = interp_func(nan_indices)  # Linha 72
        series[np.isnan(series)] = 0  # Linha 73
        X_imputed[i, :] = series  # Linha 74

    # Tratamento de Outliers (clipagem em Média + 2*Desvio Padrão)
    avg_x = np.mean(X_imputed, axis=0)  # Linha 77
    std_x = np.std(X_imputed, axis=0)  # Linha 78
    threshold_outlier = avg_x + 2 * std_x  # Linha 79
    X_outlier_handled = X_imputed.copy()  # Linha 80

    for j in range(X_outlier_handled.shape[1]):  # Linha 82
        mask = X_outlier_handled[:, j] > threshold_outlier[j]  # Linha 83
        X_outlier_handled[mask, j] = threshold_outlier[j]  # Linha 84

    # Normalização (MinMaxScaler)
    if fit_scaler:  # Linha 87
        scaler = MinMaxScaler()  # Linha 88
        X_scaled = scaler.fit_transform(X_outlier_handled)  # Linha 89
        return X_scaled, scaler  # Linha 90
    else:
        X_scaled = scaler.transform(X_outlier_handled)  # Linha 92
        return X_scaled, scaler  # Linha 93


def evaluate_model(model, X_test, y_test):  # Linha 96: Função de avaliação detalhada para o LGBM.
    """
    Avalia o modelo LGBM e calcula as métricas detalhadas (por classe e Média Macro).
    O LGBM retorna as classes preditas (0 ou 1) com .predict().
    """
    y_pred = model.predict(X_test)  # Linha 102

    # 1. Cálculo das Métricas (usando sklearn.metrics)

    # Métrica: Acurácia geral
    accuracy = accuracy_score(y_test, y_pred) * 100  # Linha 107

    # Métricas: Média Macro (average='macro')
    # A média macro trata igualmente ambas as classes, ideal para dados desbalanceados.
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100  # Linha 111
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100  # Linha 112
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100  # Linha 113

    # Métricas: Por Classe (average=None) -> [Valor Classe 0, Valor Classe 1]
    prec_per_class = precision_score(y_test, y_pred, average=None, zero_division=0) * 100  # Linha 116
    rec_per_class = recall_score(y_test, y_pred, average=None, zero_division=0) * 100  # Linha 117
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0) * 100  # Linha 118

    results = {
        'Acc': accuracy,  # Linha 120
        'Prec(avg)': prec_macro,  # Linha 121
        'Rec(avg)': rec_macro,  # Linha 122
        'F1(avg)': f1_macro,  # Linha 123
        # Resultados por classe
        'Prec(0)': prec_per_class[0],  # Linha 125 (Classe 0: Normal)
        'Prec(1)': prec_per_class[1],  # Linha 126 (Classe 1: Fraude)
        'Rec(0)': rec_per_class[0],  # Linha 127
        'Rec(1)': rec_per_class[1],  # Linha 128
        'F1(0)': f1_per_class[0],  # Linha 129
        'F1(1)': f1_per_class[1],  # Linha 130
    }
    return results  # Linha 131


def print_results(title, results):  # Linha 134
    """Imprime os resultados detalhados das métricas no formato de tabela."""
    print(f"\n--- {title} ---")  # Linha 136
    print("Métrica | Classe 0 (Normal) | Classe 1 (Fraude) | Média Macro")  # Linha 137 (Cabeçalho da Tabela)
    print("-" * 65)  # Linha 138

    # Imprime Acurácia (é um valor único)
    acc = results['Acc']  # Linha 141
    print(f"Acc.    | {acc:.2f}% | {acc:.2f}% | {acc:.2f}%")  # Linha 142

    # Imprime Precisão (Precision)
    prec_0, prec_1, prec_avg = results['Prec(0)'], results['Prec(1)'], results['Prec(avg)']  # Linha 145
    print(f"Prec.   | {prec_0:.2f}% | {prec_1:.2f}% | {prec_avg:.2f}%")  # Linha 146

    # Imprime Recall
    rec_0, rec_1, rec_avg = results['Rec(0)'], results['Rec(1)'], results['Rec(avg)']  # Linha 149
    print(f"Recall  | {rec_0:.2f}% | {rec_1:.2f}% | {rec_avg:.2f}%")  # Linha 150

    # Imprime F1-Score
    f1_0, f1_1, f1_avg = results['F1(0)'], results['F1(1)'], results['F1(avg)']  # Linha 153
    print(f"F1-Score| {f1_0:.2f}% | {f1_1:.2f}% | {f1_avg:.2f}%")  # Linha 154
    print("-" * 65)  # Linha 155


# ------------------------------------------------------------------------------
# 4. EXECUÇÃO PRINCIPAL: LGBM Classifier TREINADO com SMOTE
# ------------------------------------------------------------------------------

# 4.1 Carregar e Dividir Dados
# Carrega os dados (reais ou simulados).
X_raw, y, N_DAYS = load_sgcc_data(DATA_PATH)  # Linha 161
# Divide os dados. Stratify garante que a proporção de classes seja mantida.
X_train_full, X_test, y_train_full, y_test = train_test_split(X_raw, y, test_size=0.3, random_state=SEED,
                                                              stratify=y)  # Linha 163

print("\n" + "=" * 60)  # Linha 165
print(f"Dados de Treino (Original): {X_train_full.shape}")  # Linha 166
initial_counts = Counter(y_train_full)  # Linha 167
print(f"Distribuição de Classes no Treino: {initial_counts}")  # Linha 168
print("=" * 60)  # Linha 169

# 4.2 Pré-processamento e Normalização
# Ajusta o scaler SOMENTE nos dados de treino.
X_train_scaled, scaler = preprocess_data(X_train_full, fit_scaler=True)  # Linha 172
# Transforma os dados de teste usando o scaler ajustado.
X_test_scaled, _ = preprocess_data(X_test, fit_scaler=False, scaler=scaler)  # Linha 174

# 4.3 Aplicação do SMOTE (Apenas nos dados de TREINO)
print("Aplicando SMOTE aos dados de Treinamento...")  # Linha 177
sm = SMOTE(random_state=SEED)  # Linha 178: Instancia o SMOTE.
# Gera amostras sintéticas para a classe minoritária (fraude) até igualar a classe majoritária (normal).
X_res, y_res = sm.fit_resample(X_train_scaled, y_train_full)  # Linha 180

print("-" * 60)  # Linha 182
print("Distribuição de Classes no Treino DEPOIS do SMOTE:", Counter(y_res))  # Linha 183
print(f"Número total de amostras no treino depois do SMOTE: {len(X_res)}")  # Linha 184
print("-" * 60)  # Linha 185

# 4.4 Configuração e Treinamento do LGBM Classifier
# Parâmetros OTIMIZADOS, conforme solicitado.
# O 'class_weight' NÃO é usado, pois as classes já foram balanceadas pelo SMOTE.
LGBM_PARAMS = {  # Linha 190: Dicionário com os hiperparâmetros OTIMIZADOS.
    'objective': 'binary',  # Linha 191: Define o problema como classificação binária.
    'metric': 'binary_logloss',  # Linha 192: Métrica para otimização (perda logarítmica binária).
    'boosting_type': 'gbdt',  # Linha 193: Tipo de boosting: Gradient Boosting Decision Tree.
    'n_estimators': 300,  # Linha 194: Número de árvores (Atualizado: 300).
    'learning_rate': 0.2,  # Linha 195: Taxa de aprendizado (Atualizado: 0.2).
    'num_leaves': 70,  # Linha 196: Número máximo de folhas por árvore (Atualizado: 70).
    'max_depth': 30,  # Linha 197: Profundidade máxima da árvore (Atualizado: 30).
    'min_child_samples': 30,  # Linha 198: Mínimo de amostras em um nó filho (Atualizado: 30).
    'colsample_bytree': 0.8,  # Linha 199: Proporção de colunas amostradas ao construir cada árvore.
    'subsample': 0.8,  # Linha 200: Proporção de dados amostrados (sem substituição) para construir cada árvore.
    # Removido: 'scale_pos_weight' ou 'class_weight' não é necessário com SMOTE.
    'random_state': SEED,  # Linha 201: Semente para reprodutibilidade.
    'n_jobs': -1  # Linha 202: Usa todos os núcleos da CPU para paralelização.
}

print("\n--- LGBM (SMOTE) - Configuração ---")  # Linha 204
print("Parâmetros:", LGBM_PARAMS)  # Linha 205
print(f"Treinamento em {X_res.shape[0]} amostras BALANCEADAS.")  # Linha 206

# Treinamento: Utiliza o dataset BALANCEADO (X_res, y_res)
model_lgbm_smote = LGBMClassifier(**LGBM_PARAMS)  # Linha 209
# O treinamento é feito no conjunto balanceado SMOTE.
model_lgbm_smote.fit(X_res, y_res)  # Linha 211

print("Treinamento do LGBM (SMOTE) concluído.")  # Linha 213

# 4.5 Avaliação
# A avaliação é feita no conjunto de teste original (X_test_scaled)
results_lgbm_smote = evaluate_model(model_lgbm_smote, X_test_scaled, y_test)  # Linha 217
print_results("LGBM Classifier TREINADO com SMOTE - Resultados Detalhados no Teste", results_lgbm_smote)  # Linha 218
print("=" * 60)  # Linha 219
