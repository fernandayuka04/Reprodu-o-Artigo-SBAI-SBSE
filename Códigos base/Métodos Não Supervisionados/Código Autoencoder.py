import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers

# 1. Geração de Dados
np.random.seed(42)

# Gera 500 amostras (dados "normais")
X_normal = 0.3 * np.random.randn(500, 2)
X_normal = X_normal + 2

# 2. Inserção de Anomalias (Outliers)
n_anomalies = 20
X_anomalies = np.random.uniform(low=-4, high=4, size=(n_anomalies, 2))
X_anomalies = X_anomalies + 6

# Combina os dados (X_train usado apenas para treino, X_test para avaliação)
X_all = np.vstack([X_normal, X_anomalies])
y_true = np.array([0] * len(X_normal) + [1] * len(X_anomalies)) # Rótulos: 0=Normal, 1=Anomalia

# Para o Autoencoder, treinamos APENAS com dados normais
X_train_normal, X_val_normal = train_test_split(X_normal, test_size=0.2, random_state=42)

# 3. Pré-processamento: Normalização
# É crucial para Redes Neurais
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_normal)
X_all_scaled = scaler.transform(X_all) # Transforma todos os dados

# 4. Construção do Modelo Autoencoder
input_dim = X_train_scaled.shape[1]
latent_dim = 1 # Dimensão do gargalo (bottleneck)

# Modelo Sequencial (Encoder + Decoder)
autoencoder = keras.Sequential(
    [
        # Encoder
        layers.Dense(units=4, activation="relu", input_shape=(input_dim,)),
        layers.Dense(units=latent_dim, activation="relu", name="bottleneck"),
        # Decoder
        layers.Dense(units=4, activation="relu"),
        layers.Dense(units=input_dim, activation="sigmoid"), # Sigmoid ou linear para normalizar [0, 1]
    ]
)

autoencoder.compile(optimizer='adam', loss='mse') # Loss Mean Squared Error (MSE) para reconstrução

# 5. Treinamento
# Treina o Autoencoder usando APENAS os dados normais
history = autoencoder.fit(
    X_train_scaled,
    X_train_scaled, # X de entrada é o mesmo X de saída desejado
    epochs=50,
    batch_size=16,
    validation_data=(X_val_normal, X_val_normal),
    shuffle=True,
    verbose=0
)

# 6. Predição: Cálculo do Erro de Reconstrução
# Prediz a reconstrução para TODOS os dados (normais e anomalias)
reconstructions = autoencoder.predict(X_all_scaled, verbose=0)

# Calcula o Erro Quadrático Médio (MSE) entre a entrada original e a reconstrução
mse = np.mean(np.power(X_all_scaled - reconstructions, 2), axis=1)

# 7. Definição do Limite (Threshold)
# O limite é definido com base no erro de reconstrução dos dados NORMAIS de TREINAMENTO
# Usamos um percentil para determinar o limiar de aceitação
# 95% dos erros dos dados normais devem estar abaixo deste limite.
# Dados acima deste limite serão classificados como anomalias.
threshold = np.percentile(mse[y_true == 0], 95)

# Classificação: Anomalia (1) se o erro for > threshold, Normal (0) caso contrário
y_pred = (mse > threshold).astype(int)

# 8. Resultados em Números (Métricas de Avaliação)
# O rótulo "anomalia" (classe 1) é o positivo para as métricas.
print("## 📊 Resultados em Números")
print("-" * 30)

print(f"Total de Amostras: {len(X_all)}")
print(f"Anomalias Inseridas (Verdadeiras): {n_anomalies}")
print(f"Anomalias Detectadas (Preditas): {np.sum(y_pred == 1)}")
print(f"Limite (Threshold) de Erro MSE: {threshold:.4f}")

print("\n--- Matriz de Confusão ---")
# cm: [[TN, FP], [FN, TP]] (0=Normal, 1=Anomalia)
cm = confusion_matrix(y_true, y_pred)
print("Predito: Normal (0) | Predito: Anomalia (1)")
print(f"Real: Normal (0)   -> | {cm[0, 0]:<15}| {cm[0, 1]:<10}")
print(f"Real: Anomalia (1) -> | {cm[1, 0]:<15}| {cm[1, 1]:<10}")

print("\n--- Métricas ---")
print(f"Acurácia: {accuracy_score(y_true, y_pred):.4f}")
print(f"Precisão (Anomalia 1): {precision_score(y_true, y_pred, pos_label=1):.4f}")
print(f"Recall (Anomalia 1): {recall_score(y_true, y_pred, pos_label=1):.4f}")
print(f"F1-Score (Anomalia 1): {f1_score(y_true, y_pred, pos_label=1):.4f}")


# 9. Resultados em Gráficos
print("\n## 📈 Resultados em Gráficos")

plt.figure(figsize=(12, 5))

# Gráfico 1: Erro de Reconstrução por Amostra
plt.subplot(1, 2, 1)
plt.hist(mse[y_true == 0], bins=50, label='Normal (Classe 0)', alpha=0.6, color='blue')
plt.hist(mse[y_true == 1], bins=50, label='Anomalia (Classe 1)', alpha=0.9, color='red')
plt.axvline(threshold, color='green', linestyle='--', label=f'Threshold ({threshold:.4f})')
plt.title('Distribuição do Erro de Reconstrução (MSE)')
plt.xlabel('Erro Quadrático Médio (MSE)')
plt.ylabel('Frequência')
plt.legend()

# Gráfico 2: Detecção de Anomalias no Espaço 2D
plt.subplot(1, 2, 2)
# Plota dados normais corretamente classificados (True Negative)
plt.scatter(X_all[y_true == 0, 0], X_all[y_true == 0, 1], c='blue', edgecolors='k', s=20, label='Normal (TN/FP)')
# Plota anomalias corretamente classificadas (True Positive)
plt.scatter(X_all[(y_true == 1) & (y_pred == 1), 0], X_all[(y_true == 1) & (y_pred == 1), 1], c='red', marker='o', s=50, label='Anomalia Detectada (TP)')
# Plota Anomalias não detectadas (False Negative)
plt.scatter(X_all[(y_true == 1) & (y_pred == 0), 0], X_all[(y_true == 1) & (y_pred == 0), 1], c='red', marker='x', s=100, linewidths=2, label='Anomalia Perdida (FN)')
# Plota Falsos Positivos (Normais classificados como anomalias)
plt.scatter(X_all[(y_true == 0) & (y_pred == 1), 0], X_all[(y_true == 0) & (y_pred == 1), 1], c='orange', marker='s', s=80, linewidths=1, label='Falso Positivo (FP)')

plt.title("Detecção de Anomalias (Autoencoder)")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.legend()
plt.tight_layout()
plt.show()