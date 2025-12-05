import pandas as pd
import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# 1. Geração de Dados Pequenos e Desbalanceados
# Cria um dataset sintético com 100 amostras, 20 características e um desbalanceamento de 90/10
# (Classe 0: ~90 amostras, Classe 1: ~10 amostras)
X, y = make_classification(n_samples=100, n_features=20, n_informative=10,
                           n_redundant=10, n_classes=2, n_clusters_per_class=1,
                           weights=[0.90, 0.10], flip_y=0, random_state=42)

# Converter para DataFrame para melhor visualização
df_X = pd.DataFrame(X)
df_y = pd.Series(y)

# Contagem inicial das classes
print("## 📊 Distribuição Original das Classes")
original_count = Counter(df_y)
print(f"Classe 0 (Majoritária): {original_count[0]} amostras")
print(f"Classe 1 (Minoritária): {original_count[1]} amostras")
print("-" * 40)

# 2. Aplicação do SMOTE
# target_percent = 1.0 significa que queremos que a classe minoritária atinja
# o mesmo número de amostras da classe majoritária.
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(df_X, df_y)

# 3. Contagem Após SMOTE
print("## ✨ Distribuição das Classes Após SMOTE")
resampled_count = Counter(y_res)
print(f"Classe 0 (Majoritária): {resampled_count[0]} amostras")
print(f"Classe 1 (Minoritária - Sintetizada): {resampled_count[1]} amostras")
print("-" * 40)

# 4. Visualização (Apenas para as duas primeiras características)
plt.figure(figsize=(12, 5))

# Plot dos dados originais
plt.subplot(1, 2, 1)
plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Classe 0 (Original)', alpha=0.6, s=50)
plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Classe 1 (Original)', alpha=0.9, s=50)
plt.title('Dados Originais Desbalanceados')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Plot dos dados após SMOTE
plt.subplot(1, 2, 2)
# Pontos da Classe 0 (Majoritária)
plt.scatter(X_res[y_res == 0].iloc[:, 0], X_res[y_res == 0].iloc[:, 1], label='Classe 0 (Original)', alpha=0.6, s=50)
# Pontos da Classe 1 (Original)
plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Classe 1 (Original)', alpha=0.9, s=50)
# Pontos da Classe 1 (Sintetizados pelo SMOTE)
# Identifica os pontos que foram adicionados (diferença entre os resampled e os originais)
synthesized_indices = y_res[y_res == 1].index[len(X[y == 1]):]
plt.scatter(X_res.iloc[synthesized_indices, 0], X_res.iloc[synthesized_indices, 1],
            label='Classe 1 (SMOTE Sintetizada)', alpha=0.5, s=100, marker='x', color='red')

plt.title('Dados Após Aplicação do SMOTE (Balanceados)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.tight_layout()
plt.show()