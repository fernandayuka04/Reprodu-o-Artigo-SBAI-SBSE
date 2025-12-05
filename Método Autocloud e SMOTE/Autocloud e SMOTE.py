# ==============================================================================
# 1. Importação das Bibliotecas Necessárias
# ==============================================================================
import pandas as pd # Linha 4: Importa a biblioteca pandas para manipulação de dados em DataFrames.
import numpy as np  # Linha 5: Importa a biblioteca numpy para operações numéricas e geração de dados.
from collections import Counter # Linha 6: Importa a classe Counter para contar a frequência das classes.
from imblearn.over_sampling import SMOTE # Linha 7: Importa a classe SMOTE, que realiza o oversampling sintético.
from sklearn.preprocessing import StandardScaler # Linha 8: Importa o StandardScaler para normalizar os dados.
from tensorflow.keras.models import Model # Linha 9: Importa a classe Model para construir o Autoencoder.
from tensorflow.keras.layers import Input, Dense # Linha 10: Importa as camadas Input e Dense para a rede neural.
from tensorflow.keras.losses import MeanSquaredError # Linha 11: Importa a função de perda MSE.
# Linha 12: Importa as métricas individuais para cálculo detalhado (por classe e média).
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# O pipeline será: Simulação -> Normalização -> SMOTE -> Autoencoder.

# ==============================================================================
# 2. Simulação de um Conjunto de Dados Desequilibrado (Contexto Autocloud/ETD)
# ==============================================================================

# Definição do número de amostras normais e anômalas.
n_normal = 10000 # 10.000 registros normais (classe majoritária).
n_anomaly = 100  # 100 registros de fraude/anomalia (classe minoritária).
n_features = 5 # Número de características (features) de consumo.

# Geração de dados (Features de consumo, por exemplo, consumo médio, desvio, etc.)
# Dados Normais (distribuição normal)
X_normal = np.random.normal(loc=100, scale=10, size=(n_normal, n_features)) # Gera dados normais.
y_normal = np.zeros(n_normal, dtype=int) # Rótulo 0 (Normal).

# Dados Anômalos (distribuição diferente para simular fraude/perda não técnica)
X_anomaly = np.random.normal(loc=150, scale=25, size=(n_anomaly, n_features)) # Gera dados anômalos.
y_anomaly = np.ones(n_anomaly, dtype=int) # Rótulo 1 (Anomalia/Fraude).

# Combinação e Preparação Final dos Dados
X = np.vstack((X_normal, X_anomaly)) # Combina as features.
y = np.concatenate((y_normal, y_anomaly)) # Combina os rótulos.
df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(n_features)]) # Cria o DataFrame.
df['Target'] = y # Adiciona o rótulo.
X = df.drop('Target', axis=1) # Separa X.
y = df['Target'] # Separa y.

# ==============================================================================
# 3. Pré-processamento e SMOTE
# ==============================================================================

# 3.1 Normalização dos dados
scaler = StandardScaler() # Instancia o objeto StandardScaler.
X_scaled = scaler.fit_transform(X) # Normaliza os dados.
X_scaled = pd.DataFrame(X_scaled, columns=X.columns) # Converte para DataFrame.

print("=" * 60) # Imprime linha separadora.
print("Distribuição das Classes ANTES do SMOTE:", Counter(y)) # Mostra o desequilíbrio.

# 3.2 Aplicação do SMOTE
sm = SMOTE(random_state=42) # Instancia o SMOTE.
# Aplica o SMOTE no conjunto de dados normalizado.
X_res, y_res = sm.fit_resample(X_scaled, y) # Geração de amostras sintéticas e reamostragem.

print("-" * 60) # Imprime linha separadora.
print("Distribuição das Classes DEPOIS do SMOTE:", Counter(y_res)) # Mostra o equilíbrio.
print(f"Número total de amostras depois do SMOTE: {len(X_res)}") # Novo tamanho do dataset.
print("=" * 60) # Imprime linha separadora.

# ==============================================================================
# 4. Construção e Treinamento do Autoencoder (Autocloud Simulado)
# ==============================================================================

# 4.1 Definição da Arquitetura do Autoencoder
input_layer = Input(shape=(n_features,)) # Define a camada de entrada com 5 features.
# Camada Encoder (Compactação de 5 para 3 features)
encoder = Dense(3, activation='relu')(input_layer) # Camada densa com 3 neurônios e ativação ReLU.
# Camada Decoder (Reconstrução de 3 para 5 features)
decoder = Dense(n_features, activation='linear')(encoder) # Camada de saída com 5 neurônios (original) e ativação linear.

# Criação do Modelo
autoencoder = Model(inputs=input_layer, outputs=decoder) # Define o modelo completo (entrada -> saída).

# 4.2 Compilação do Modelo
autoencoder.compile(optimizer='adam', loss=MeanSquaredError()) # Usa o otimizador Adam e a perda MSE (Erro Quadrático Médio).

print("Treinando o Autoencoder com os dados balanceados (X_res)...") # Informa o início do treinamento.

# 4.3 Treinamento do Modelo
# X_res é usado tanto como entrada (input) quanto como saída (target) no treinamento do AE.
autoencoder.fit(X_res, X_res, # Entrada e Saída são as próprias features reamostradas.
                epochs=100, # Número de épocas de treinamento. *** ALTERADO PARA 100 ÉPOCAS ***
                batch_size=32, # Tamanho do lote.
                shuffle=True, # Embaralha os dados em cada época.
                verbose=1) # Altera de 0 para 1 para exibir o progresso do treinamento.

print("Treinamento concluído.") # Informa a conclusão.

# ==============================================================================
# 5. Funções de Avaliação Detalhada (Inclusão)
# ==============================================================================

def evaluate_model_detailed(Y_test, Y_pred):
    """
    Linha 127: Calcula Acurácia, Precisão, Recall e F1-Score por classe e Média Macro.
    """
    # Métrica: Acurácia geral
    accuracy = accuracy_score(Y_test, Y_pred) * 100 # Linha 132

    # Média Macro (average='macro') - VALOR MÉDIO
    # Linha 135-137: Cálculo dos valores médios.
    prec_macro = precision_score(Y_test, Y_pred, average='macro', zero_division=0) * 100
    rec_macro = recall_score(Y_test, Y_pred, average='macro', zero_division=0) * 100
    f1_macro = f1_score(Y_test, Y_pred, average='macro', zero_division=0) * 100

    # Por Classe (average=None) - Base para MENOR e MAIOR valor
    # Linha 140-142: Cálculo dos valores individuais por classe.
    prec_per_class = precision_score(Y_test, Y_pred, average=None, zero_division=0) * 100
    rec_per_class = recall_score(Y_test, Y_pred, average=None, zero_division=0) * 100
    f1_per_class = f1_score(Y_test, Y_pred, average=None, zero_division=0) * 100

    results = {
        'Acc': accuracy,
        'Prec(avg)': prec_macro,
        'Rec(avg)': rec_macro,
        'F1(avg)': f1_macro,
        # Linha 149-154: Resultados por classe (Classe 0: Normal, Classe 1: Anomalia).
        'Prec(0)': prec_per_class[0],
        'Prec(1)': prec_per_class[1],
        'Rec(0)': rec_per_class[0],
        'Rec(1)': rec_per_class[1],
        'F1(0)': f1_per_class[0],
        'F1(1)': f1_per_class[1],
    }
    return results


def print_detailed_results(title, results):
    """Linha 159: Imprime os resultados detalhados das métricas no formato de tabela."""
    print(f"\n--- {title} ---")
    # Linha 162: Cabeçalho da Tabela
    print("Métrica | Classe 0 (Normal) | Classe 1 (Anomalia) | Média Macro")
    print("-" * 65)

    # Imprime Acurácia (valor único)
    acc = results['Acc']
    print(f"Acc.    | {acc:.2f}% | {acc:.2f}% | {acc:.2f}%")

    # Imprime Precisão
    prec_0, prec_1, prec_avg = results['Prec(0)'], results['Prec(1)'], results['Prec(avg)']
    print(f"Prec.   | {prec_0:.2f}% | {prec_1:.2f}% | {prec_avg:.2f}%")

    # Imprime Recall
    rec_0, rec_1, rec_avg = results['Rec(0)'], results['Rec(1)'], results['Rec(avg)']
    print(f"Recall  | {rec_0:.2f}% | {rec_1:.2f}% | {rec_avg:.2f}%")

    # Imprime F1-Score
    f1_0, f1_1, f1_avg = results['F1(0)'], results['F1(1)'], results['F1(avg)']
    print(f"F1-Score| {f1_0:.2f}% | {f1_1:.2f}% | {f1_avg:.2f}%")
    print("-" * 65)


# ==============================================================================
# 6. Avaliação do Autoencoder e Detecção de Anomalias
# ==============================================================================

# 6.1 Reconstrução (Aplicação do modelo aos dados originais)
X_predictions = autoencoder.predict(X_scaled, verbose=0) # Linha 190: Faz a previsão/reconstrução para todas as amostras originais normalizadas.

# 6.2 Cálculo do Erro de Reconstrução (MSE)
mse = np.mean(np.power(X_scaled - X_predictions, 2), axis=1) # Linha 193: Calcula o MSE de cada amostra.

# 6.3 Definição do Limiar (Threshold)
error_normal = mse[y == 0] # Linha 196: Filtra os erros de reconstrução apenas para as amostras "Normais" originais.
threshold = np.mean(error_normal) + 2 * np.std(error_normal) # Linha 197: Limiar = Média + 2 * Desvio Padrão.

print("-" * 60) # Imprime linha separadora.
print(f"Limiar de Erro de Reconstrução (Threshold): {threshold:.4f}") # Exibe o limiar calculado.

# 6.4 Classificação e Resultados
# Se o erro de reconstrução for MAIOR que o limiar, classifica-se como Anomalia (1).
y_pred = (mse > threshold).astype(int) # Linha 203: Cria as previsões binárias (0 ou 1).

print("-" * 60) # Imprime linha separadora.

# Linha 206-207: Chamada às funções de avaliação detalhada.
detailed_results = evaluate_model_detailed(y, y_pred)
print_detailed_results("Métricas Detalhadas - Autoencoder com SMOTE", detailed_results)

print("=" * 60) # Imprime linha separadora.
