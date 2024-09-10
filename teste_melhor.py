from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
teste = 0.7
# Função centralizada para carregar e dividir os dados, com normalização
def load_and_split_data(test_size=0.3, random_state=42):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=test_size, random_state=random_state)
    
    # Normalização dos dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Função para classificação utilizando MLP
def mlp_classification(hidden_layer_sizes=(10, 10), activation='relu', solver='adam', max_iter=214, alpha=0.0001, test_size=0.3):
    X_train, X_test, y_train, y_test = load_and_split_data(test_size=test_size)
    
    # Criando o modelo MLP
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, max_iter=max_iter, alpha=alpha, random_state=42)
    
    # Treinando o modelo e testando
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    
    print("Resultados MLP:")
    print("Acurácia:", 100 * accuracy_score(y_test, y_pred))

# Função para classificação utilizando Decision Tree
def decision_tree_classification(max_depth=None, min_samples_split=2, test_size=0.3):
    X_train, X_test, y_train, y_test = load_and_split_data(test_size=test_size)
    
    # Criando o modelo Decision Tree
    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    
    # Treinando o modelo e testando
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    
    print("Resultados Decision Tree:")
    print("Acurácia:", 100 * accuracy_score(y_test, y_pred))

# Função para classificação utilizando SVM
def svm_classification(C=1.0, kernel='linear', gamma='scale', test_size=0.3):
    X_train, X_test, y_train, y_test = load_and_split_data(test_size=test_size)
    
    # Criando o modelo SVM
    svm = SVC(C=C, kernel=kernel, gamma=gamma)
    
    # Treinando o modelo e testando
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    
    print("Resultados SVM:")
    print("Acurácia:", 100 * accuracy_score(y_test, y_pred))

# Função para classificação utilizando KNN
def knn_classification(n_neighbors=3, metric='euclidean', weights='uniform', test_size=0.3):
    X_train, X_test, y_train, y_test = load_and_split_data(test_size=test_size)
    
    # Criando o modelo KNN
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
    
    # Treinando o modelo e testando
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    print("Resultados KNN:")
    print("Acurácia:", 100 * accuracy_score(y_test, y_pred))

# Chamadas de função para testar os classificadores
mlp_classification(test_size=teste)  # Exemplo com 60% treino e 40% teste
decision_tree_classification(test_size=teste)
svm_classification(test_size=teste)
knn_classification(test_size=teste)
