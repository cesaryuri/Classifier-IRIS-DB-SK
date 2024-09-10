from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


# Função de carregamento e divisão de dados com normalização
def load_and_split_data(test_size=0.3, random_state=42):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=test_size, random_state=random_state)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def arvore_de_decisao(max_depth=3, min_samples_split=2):
    # recebendo os dados já dividos prontos e normalizados
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    # Criando o modelo
    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    
    # Treinando o modelo e em seguida testando
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print("Resultados Árvore de decisão:")
    print("Acurácia:", 100*accuracy_score(y_test, y_pred))


arvore_de_decisao()