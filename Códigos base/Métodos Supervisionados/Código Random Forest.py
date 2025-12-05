# 1. Importar as bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Preparação dos Dados (Exemplo com o dataset Iris) ---

# 2. Carregar ou criar um dataset (aqui, vamos criar um dataset simples para o exemplo)
# Na prática, você carregaria seus dados com pd.read_csv('seu_arquivo.csv')
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data   # Variáveis independentes (features)
y = iris.target # Variável dependente (rótulos/classes)

# 3. Criar um DataFrame para visualização (opcional, mas bom para entender os dados)
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# 4. Dividir os dados em conjuntos de Treinamento e Teste
# X_train, y_train: Usados para treinar o modelo
# X_test, y_test: Usados para avaliar o desempenho do modelo
X_train, X_test, y_train, y_test = train_test_split(
    X,                      # O conjunto completo de features
    y,                      # Os rótulos/targets correspondentes
    test_size=0.3,          # 30% dos dados serão usados para teste
    random_state=42         # Garante que a divisão seja a mesma em execuções diferentes
)

# --- Construção e Treinamento do Modelo Random Forest ---

# 5. Inicializar o modelo Random Forest Classifier
# n_estimators: Número de árvores de decisão na floresta (mais árvores = geralmente melhor, mas mais lento)
# random_state: Para reprodutibilidade dos resultados
model = RandomForestClassifier(n_estimators=100, random_state=42)


# 6. Treinar o modelo
# O modelo aprende os padrões a partir dos dados de treinamento (X_train, y_train)
model.fit(X_train, y_train)

# --- Avaliação do Modelo ---

# 7. Fazer previsões no conjunto de teste
# O modelo faz previsões sobre os dados que ele nunca viu (X_test)
y_pred = model.predict(X_test)

# 8. Avaliar o desempenho do modelo (calculando a acurácia)
# A acurácia é a proporção de previsões corretas
accuracy = accuracy_score(y_test, y_pred)

# 9. Imprimir os resultados
print(f"Número de árvores na floresta (n_estimators): {model.get_params()['n_estimators']}")
print(f"Acurácia do modelo Random Forest: {accuracy:.4f}") # Formata a acurácia com 4 casas decimais

# --- Análise Adicional (Importância dos Atributos) ---

# 10. Obter a importância de cada feature (atributo)
# Random Forest pode dizer quais features foram mais importantes para as previsões
feature_importances = pd.Series(
    model.feature_importances_,  # Os valores de importância calculados pelo modelo
    index=iris.feature_names     # Os nomes das features
).sort_values(ascending=False)   # Classifica do mais importante para o menos importante

# 11. Imprimir a importância dos atributos
print("\nImportância dos Atributos:")
print(feature_importances)

# --- Exemplo de Nova Previsão ---

# 12. Fazer uma previsão em um novo dado
# Exemplo: um novo dado com as 4 features
novo_dado = [[5.1, 3.5, 1.4, 0.2]] # Valores de Sepal Length, Sepal Width, Petal Length, Petal Width
previsao_nova = model.predict(novo_dado)

# 13. Mapear o resultado da previsão de volta para o nome da classe (para clareza)
nome_classe = iris.target_names[previsao_nova[0]]

# 14. Imprimir a previsão para o novo dado
print(f"\nPrevisão para o novo dado {novo_dado}: Classe {previsao_nova[0]} ({nome_classe})")
