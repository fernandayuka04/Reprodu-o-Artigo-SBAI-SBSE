import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
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

# 3. Treinamento do Modelo Isolation Forest
# O parâmetro 'contamination' é a fração esperada de outliers no conjunto de dados.
contamination_rate = n_anomalies / n_samples

# n_estimators: número de árvores na floresta (mais árvores geralmente melhoram a precisão)
# max_samples: número de amostras a serem desenhadas para treinar cada árvore
model = IsolationForest(contamination=contamination_rate, random_state=42, n_estimators=100)

# Treina o modelo (não supervisionado, usa apenas os dados)
model.fit(X)

# 4. Predição e Avaliação
# A predição retorna 1 para pontos inliers (normais) e -1 para pontos outliers (anomalias)
y_pred = model.predict(X)

# --- CORREÇÃO: Cálculo do Limiar de Decisão para Plotagem ---
# O limiar não é acessível diretamente via 'threshold_'.
# Calculamos o limiar a partir dos scores da função de decisão (decision_function)
# no conjunto de treino (X) e na taxa de contaminação (contamination_rate).
scores = model.decision_function(X)
# O limiar é o score no percentil correspondente à taxa de contaminação.
# Pontos com scores ABAIXO deste valor são classificados como anomalias.
model_threshold = np.percentile(scores, contamination_rate * 100)
# -----------------------------------------------------------------


# 5. Resultados em Números (Métricas de Avaliação)
print("## 📊 Resultados em Números")
print("-" * 30)

# Matriz de Confusão e Cálculo de Métricas (usando o rótulo -1 como positivo para anomalia)
print(f"Total de Amostras: {n_samples}")
print(f"Anomalias Inseridas (Verdadeiras): {n_anomalies}")
print(f"Anomalias Detectadas (Preditas): {list(y_pred).count(-1)}")
print("\n--- Matriz de Confusão ---")
print("Predito: Anomalia (-1) | Predito: Normal (1)")
# Conta True Positive e False Negative
tp = list(y_pred[y_true == -1]).count(-1)
fn = list(y_pred[y_true == -1]).count(1)
# Conta False Positive e True Negative
fp = list(y_pred[y_true == 1]).count(-1)
tn = list(y_pred[y_true == 1]).count(1)

print(f"Real: Anomalia (-1) -> | {tp:<15}| {fn:<10}")
print(f"Real: Normal (1)   -> | {fp:<15}| {tn:<10}")
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

# Calcula a função de decisão (scores) para cada ponto na malha
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))

# Desenha o contorno de decisão
# O valor de corte é determinado pelo modelo baseado no 'contamination'
plt.contourf(xx, yy, Z, cmap=plt.cm.YlGnBu, alpha=0.3)
# Linha de fronteira (limite entre inlier e outlier) - USANDO O LIMIAR CALCULADO
plt.contour(xx, yy, Z, levels=[model_threshold], linewidths=2, colors='darkgreen')

# Plota os dados normais
plt.scatter(X_normal[:, 0], X_normal[:, 1], c='white', edgecolors='k', s=20, label='Dados Normais (Verdadeiros)')
# Plota as anomalias (verdadeiras)
plt.scatter(X_anomalies[:, 0], X_anomalies[:, 1], c='red', edgecolors='k', s=50, label='Anomalias (Verdadeiras)')

# Marca os pontos classificados como anomalia (-1)
anomalies_pred_idx = np.where(y_pred == -1)
plt.scatter(X[anomalies_pred_idx, 0], X[anomalies_pred_idx, 1], c='orange', marker='x', s=100, linewidths=2, label='Anomalias (Preditas)')

plt.title("Isolation Forest para Detecção de Anomalias")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.legend()
plt.show()