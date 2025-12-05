# ==============================================================================
# 1. Importação das Bibliotecas Necessárias
# ==============================================================================
import pandas as pd # Importa a biblioteca pandas para manipulação de dados em DataFrames.
import numpy as np  # Importa a biblioteca numpy para operações numéricas e geração de dados.
from collections import Counter # Importa a classe Counter para contar a frequência das classes.
from imblearn.over_sampling import SMOTE # Importa a classe SMOTE, que realiza o oversampling sintético.
from sklearn.preprocessing import StandardScaler # Importa o StandardScaler para normalizar os dados.
from tensorflow.keras.models import Model # Importa a classe Model para construir o Autoencoder.
from tensorflow.keras.layers import Input, Dense # Importa as camadas Input e Dense para a rede neural.
from tensorflow.keras.losses import MeanSquaredError # Importa a função de perda MSE.
from sklearn.metrics import classification_report # Importa métricas para avaliar o resultado.

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

# Nota: O uso do Autoencoder após o SMOTE é para fins de exemplo de pipeline.
# Na prática de *detecção de anomalias* pura, o AE é geralmente treinado
# SOMENTE com dados da classe Normal (X_res[y_res == 0]).

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
                epochs=10, # Número de épocas de treinamento.
                batch_size=32, # Tamanho do lote.
                shuffle=True, # Embaralha os dados em cada época.
                verbose=0) # Não exibe o progresso do treinamento para manter a saída limpa.

print("Treinamento concluído.") # Informa a conclusão.

# ==============================================================================
# 5. Avaliação do Autoencoder e Detecção de Anomalias
# ==============================================================================

# 5.1 Reconstrução (Aplicação do modelo aos dados originais)
# Para a avaliação, usamos o conjunto de dados ORIGINAL (X_scaled) para testar a generalização.
X_predictions = autoencoder.predict(X_scaled, verbose=0) # Faz a previsão/reconstrução para todas as amostras originais.

# 5.2 Cálculo do Erro de Reconstrução (MSE)
mse = np.mean(np.power(X_scaled - X_predictions, 2), axis=1) # Calcula o MSE de cada amostra: (real - reconstruído)^2.

# 5.3 Definição do Limiar (Threshold)
# No método clássico, o limiar é definido com base no erro das amostras normais.
# Aqui, usaremos um limiar simples baseado na média + 2 desvios padrão do erro das amostras normais.
error_normal = mse[y == 0] # Filtra os erros de reconstrução apenas para as amostras "Normais" originais.
threshold = np.mean(error_normal) + 2 * np.std(error_normal) # Limiar = Média + 2 * Desvio Padrão.

print("-" * 60) # Imprime linha separadora.
print(f"Limiar de Erro de Reconstrução (Threshold): {threshold:.4f}") # Exibe o limiar calculado.

# 5.4 Classificação e Resultados
# Se o erro de reconstrução for MAIOR que o limiar, classifica-se como Anomalia (1).
y_pred = (mse > threshold).astype(int) # Cria as previsões binárias (0 ou 1).

print("-" * 60) # Imprime linha separadora.
print("Relatório de Classificação nos Dados Originais (Com 100 Anomalias):")
# Gera e exibe o relatório de classificação comparando os rótulos reais (y) com as previsões (y_pred).
print(classification_report(y, y_pred, target_names=['Normal (0)', 'Anomalia (1)']))
print("=" * 60) # Imprime linha separadora.

# Resumo da Lógica do Autoencoder: O Autoencoder tenta reconstruir a entrada.
# Se for treinado com o dataset balanceado pelo SMOTE (que agora contém anomalias sintéticas),
# ele aprende a reconstruir tanto o padrão normal quanto o padrão de anomalia. O erro
# de reconstrução será alto apenas para amostras que estão *muito* fora do padrão.
# Se o AE fosse treinado *apenas* com a classe 0 (Normal), o erro de reconstrução seria
# sempre muito alto para qualquer amostra de Anomalia, o que é o método clássico.