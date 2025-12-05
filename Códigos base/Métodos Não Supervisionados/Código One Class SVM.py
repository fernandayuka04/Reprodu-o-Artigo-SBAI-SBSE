import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 1. Geração de Dados
# Define a semente aleatória para reprodutibilidade
np.random.seed(42)

# Gera 500 amostras (dados "normais") a partir de uma distribuição normal 2D
X_normal = 0.3 * np.random.randn(500, 2)
# Desloca o centro da distribuição para (2, 2)
X_normal = X_normal + 2

# 2. Inserção de Anomalias (Outliers)
# Gera 20 anomalias bem distantes dos dados normais
n_anomalies = 20
X_anomalies = np.random.uniform(low=-4, high=4, size=(n_anomalies, 2))
X_anomalies = X_anomalies + 6 # Garante que as anomalias fiquem longe

# Combina os dados normais e as anomalias
X = np.vstack([X_normal, X_anomalies])
n_samples = len(X)

# Cria os rótulos verdadeiros: 1 para normal, -1 para anomalia
# 500 normais (1) e 20 anomalias (-1)
y_true = np.array([1] * len(X_normal) + [-1] * len(X_anomalies))

# 3. Treinamento do Modelo One-Class SVM
# O parâmetro 'nu' (nível de contaminação) é o limite superior para a fração de outliers
# e um limite inferior para a fração de vetores de suporte.
# Aqui, definimos 'nu' com base na taxa de anomalias inseridas (20 / 520 ≈ 0.038)
nu_rate = n_anomalies / n_samples
model = OneClassSVM(kernel='rbf', gamma='auto', nu=nu_rate)

# Treina o modelo usando apenas os dados (sem rótulos)
# O OCSVM assume que todos os dados de treinamento são 'normais'
model.fit(X)

# 4. Predição e Avaliação
# A predição retorna 1 para pontos dentro do limite (normais) e -1 para pontos fora (anomalias)
y_pred = model.predict(X)

# 5. Resultados em Números (Métricas de Avaliação)
print("## 📊 Resultados em Números")
print("-" * 30)

# Matriz de Confusão
cm = confusion_matrix(y_true, y_pred)
# Reordena para Anomalia(-1) / Normal(1)
cm_reordered = np.array([[cm[1, 1], cm[1, 0]], [cm[0, 1], cm[0, 0]]])
# TN FP
# FN TP
# True Negative (Anomalia Corretamente Identificada)
tn = cm[0, 0] if y_true[0] == -1 else cm[1, 1]
# True Positive (Normal Corretamente Identificado)
tp = cm[1, 1] if y_true[0] == 1 else cm[0, 0]
# False Positive (Normal Identificado como Anomalia)
fp = cm[0, 1] if y_true[0] == 1 else cm[1, 0]
# False Negative (Anomalia Identificada como Normal)
fn = cm[1, 0] if y_true[0] == 1 else cm[0, 1]

print(f"Total de Amostras: {n_samples}")
print(f"Anomalias Inseridas (Verdadeiras): {n_anomalies}")
print(f"Anomalias Detectadas (Preditas): {list(y_pred).count(-1)}")
print("\n--- Matriz de Confusão ---")
print("Predito: Anomalia (-1) | Predito: Normal (1)")
print(f"Real: Anomalia (-1) -> | {list(y_pred[y_true == -1]).count(-1):<15}| {list(y_pred[y_true == -1]).count(1):<10}")
print(f"Real: Normal (1)   -> | {list(y_pred[y_true == 1]).count(-1):<15}| {list(y_pred[y_true == 1]).count(1):<10}")
print("\n--- Métricas ---")
print(f"Acurácia: {accuracy_score(y_true, y_pred):.4f}")
print(f"Precisão (Anomalia -1): {precision_score(y_true, y_pred, pos_label=-1):.4f}")
print(f"Recall (Anomalia -1): {recall_score(y_true, y_pred, pos_label=-1):.4f}")
print(f"F1-Score (Anomalia -1): {f1_score(y_true, y_pred, pos_label=-1):.4f}")


# 6. Resultados em Gráficos
print("\n## 📈 Resultados em Gráficos")
# Cria uma malha para desenhar o contorno de decisão
xx, yy = np.meshgrid(np.linspace(min(X[:, 0])-1, max(X[:, 0])+1, 100),
                     np.linspace(min(X[:, 1])-1, max(X[:, 1])+1, 100))

# Calcula a distância do limite de decisão para cada ponto na malha
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))

# Desenha o contorno de decisão (limite entre normal e anomalia)
# 'levels=0' define a linha de fronteira
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu, alpha=0.3)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred', alpha=0.3)

# Plota os dados normais
plt.scatter(X_normal[:, 0], X_normal[:, 1], c='white', edgecolors='k', s=20, label='Dados Normais (Verdadeiros)')
# Plota as anomalias (verdadeiras)
plt.scatter(X_anomalies[:, 0], X_anomalies[:, 1], c='red', edgecolors='k', s=50, label='Anomalias (Verdadeiras)')

# Marca os pontos classificados como anomalia (-1)
anomalies_pred_idx = np.where(y_pred == -1)
plt.scatter(X[anomalies_pred_idx, 0], X[anomalies_pred_idx, 1], c='gold', marker='x', s=100, linewidths=2, label='Anomalias (Preditas)')

plt.title("One-Class SVM para Detecção de Anomalias")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.legend()
plt.show()