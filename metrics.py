import numpy as np
from ucimlrepo import fetch_ucirepo

# 1. Carregar o dataset Iris
iris = fetch_ucirepo(id=53)  # ID 53 corresponde ao dataset Iris
X = iris.data.features.to_numpy()  # Features (características)
y = iris.data.targets.to_numpy().flatten()  # Labels (rótulos)

# 2. Converter rótulos para inteiros (se forem strings)
print("Tipo dos rótulos (y):", y.dtype)

if y.dtype == 'O':  # 'O' significa objeto (geralmente strings)
    classes_unicas = np.unique(y)
    mapa_classes = {classe: idx for idx, classe in enumerate(classes_unicas)}
    y = np.array([mapa_classes[classe] for classe in y])

# 3. Função para calcular a distância euclidiana
def distancia_euclidiana(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# 4. Implementação do KNN
def knn(X_train, y_train, X_test, k):
    predicoes = []
    for x in X_test:
        distancias = [distancia_euclidiana(x, x_train) for x_train in X_train]
        k_indices = np.argsort(distancias)[:k]
        k_labels = y_train[k_indices]
        predicao = np.bincount(k_labels).argmax()
        predicoes.append(predicao)
    return np.array(predicoes)

# 5. Dividir os dados em treino, validação e teste (holdout simples)
def holdout_simples(X, y, train_size=0.7, val_size=0.15, test_size=0.15):
    # Verificar se a soma dos tamanhos é igual a 1
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("A soma de train_size, val_size e test_size deve ser igual a 1.")
    
    # Embaralhar os dados
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    
    # Definir os tamanhos dos conjuntos
    n = len(X)
    n_train = int(n * train_size)
    n_val = int(n * val_size)
    n_test = int(n * test_size)
    
    # Dividir os dados
    X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:n_train+n_val+n_test]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:n_train+n_val+n_test]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# 6. Dividindo os dados
X_train, X_val, X_test, y_train, y_val, y_test = holdout_simples(X, y, train_size=0.7, val_size=0.15, test_size=0.15)

# 7. Treinando e monitorando o aprendizado
melhor_acuracia = 0
melhor_k = 1

for k in range(1, 10):  # Testando diferentes valores de k
    y_pred_val = knn(X_train, y_train, X_val, k)
    acuracia = np.mean(y_pred_val == y_val)
    print(f"k = {k}, Acurácia na validação: {acuracia:.2f}")
    
    if acuracia > melhor_acuracia:
        melhor_acuracia = acuracia
        melhor_k = k

print(f"Melhor k: {melhor_k} com acurácia de {melhor_acuracia:.2f} na validação")

# 8. Avaliando o modelo final no conjunto de teste
y_pred_test = knn(X_train, y_train, X_test, melhor_k)
acuracia_test = np.mean(y_pred_test == y_test)
print(f"Acurácia no teste: {acuracia_test:.2f}")