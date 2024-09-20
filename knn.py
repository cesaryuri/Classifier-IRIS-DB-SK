from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# Função de carregamento e divisão de dados com normalização
def load_and_split_data(test_size=0.3, random_state=42):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=test_size, random_state=random_state)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def knn_classification(n_neighbors=3, metric='euclidean', weights='uniform'):
    # recebendo os dados já dividos prontos e normalizados
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    # Criando o modelo
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
    
    # Treinando o modelo e em seguida testando
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("Resultados KNN:")
    print("Acurácia:", 100*accuracy_score(y_test, y_pred))
    

knn_classification()