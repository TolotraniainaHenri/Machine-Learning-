
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# 1. Creation des donnees
X, y = make_regression(
    n_samples=100,
    n_features=2,
    noise=10,
    random_state=42
)

# Transformer y en colonne
y = y.reshape(-1, 1)

print("X :", X.shape)
print("y :", y.shape)

# 2. Ajouter la colonne de biais
ones = np.ones((X.shape[0], 1))

X = np.hstack((X, ones))

print("Nouvelle forme X :", X.shape)

# 3. Initialisation de theta
theta = np.random.randn(3, 1)

print("Theta initial :\n", theta)

# 4. Modele
def modele(X, theta):
    return X @ theta

# 5. Fonction de cout
def cout(X, theta, y):
    m = len(y)
    return (1 / (2 * m)) * np.sum((modele(X, theta) - y)**2)

# 6. Gradient
def gradient(X, theta, y):
    m = len(y)
    return (1 / m) * X.T @ (modele(X, theta) - y)

# 7. Descente de gradient
def descente_gradient(X, theta, alpha, y, n_iteration):
    historique = []

    for i in range(n_iteration):
        theta = theta - alpha * gradient(X, theta, y)
        historique.append(cout(X, theta, y))

    return theta, historique

# 8. Entrainement
theta_final, historique = descente_gradient(
    X, theta, 0.01, y, 1000
)

print("Theta final :\n", theta_final)

print("Cout initial :", cout(X, theta, y))
print("Cout final :", cout(X, theta_final, y))

# 9. Visualiser le cout
plt.plot(historique)
plt.xlabel("Iterations")
plt.ylabel("Cout")
plt.title("Evolution du cout")
plt.show()
