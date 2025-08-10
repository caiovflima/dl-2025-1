import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class LogisticNeuron:
    def __init__(self, input_dim, learning_rate=0.1, epochs=1000):
        self.weights = np.random.randn(input_dim)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []

    def tanh(self, z):
        # Função de ativação tanh
        return np.tanh(z)

    def tanh_derivative(self, z):
        # Derivada de tanh(z)
        return 1 - np.tanh(z)**2

    def predict_proba(self, X):
        # Saída contínua entre -1 e +1
        z = np.dot(X, self.weights) + self.bias
        a = self.tanh(z)
        return a

    def predict(self, X):
        # Classe 0 se saída < 0, classe 1 caso contrário
        return (self.predict_proba(X) >= 0).astype(int)

    def train(self, X, y):
        # Converte rótulos {0,1} para {-1,+1}
        y_tanh = 2*y - 1
        m = X.shape[0]

        for _ in range(self.epochs):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.tanh(z)

            # Erro
            error = y_pred - y_tanh

            # Gradientes para MSE + tanh
            grad_w = (1/m) * np.dot(X.T, error * self.tanh_derivative(z))
            grad_b = (1/m) * np.sum(error * self.tanh_derivative(z))

            # Atualização dos parâmetros
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            # Perda MSE
            loss = np.mean(error**2)
            self.loss_history.append(loss)

def generate_dataset():
    X, y = make_blobs(n_samples=200, centers=2, random_state=42, cluster_std=2.0)
    return X, y

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.7)
    plt.colorbar(label='Tanh Output')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.title('Tanh Activation Decision Boundary')
    plt.show()

def plot_loss(model):
    plt.plot(model.loss_history, 'k-')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('MSE Loss over Training Iterations')
    plt.show()

def main():
    X, y = generate_dataset()
    neuron = LogisticNeuron(input_dim=2, learning_rate=0.1, epochs=100)
    neuron.train(X, y)
    plot_decision_boundary(neuron, X, y)
    plot_loss(neuron)

if __name__ == "__main__":
    main()
