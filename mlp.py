from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


random_state_val = 42

# Função de carregamento e divisão de dados com normalização
def load_and_split_data(test_size=0.3, random_state=random_state_val):
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def mlp_classification(hidden_layer_sizes=(10, 10), activation='relu', solver='adam', max_iter=214, alpha=0.0001):
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    # Criando o modelo
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, max_iter=max_iter, alpha=alpha, random_state=random_state_val)
    
    # Treinando o modelo e em seguida testando
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    
    print("Resultados MLP:")
    print("Acurácia:", 100 * accuracy_score(y_test, y_pred))
    

mlp_classification()