
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# ==================================
# ETAPE 1 : CREATION DES DONNEES
# ==================================

x, y = make_regression(
    n_samples=100,
    n_features=1,
    noise=10,
    random_state=42
)

# Transformer y en matrice colonne
y = y.reshape(-1, 1)

print("Shape de x :", x.shape)
print("Shape de y :", y.shape)

# Afficher les données
plt.scatter(x, y)
plt.title("Données d'entraînement")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# ==================================
# ETAPE 2 : CREATION DE LA MATRICE X
# ==================================

ones = np.ones((x.shape[0], 1))

X = np.hstack((x, ones))

print("Shape de X :", X.shape)


# ==================================
# ETAPE 3 : INITIALISATION THETA
# ==================================

theta = np.random.randn(2, 1)

print("Theta initial :", theta)


# ==================================
# ETAPE 4 : MODELE LINEAIRE
# ==================================

def modele(X, theta):
    return X @ theta


# ==================================
# ETAPE 5 : FONCTION DE COUT
# ==================================

def cout(X, theta, y):
    m = len(y)
    predictions = modele(X, theta)
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)


# ==================================
# ETAPE 6 : CALCUL DU GRADIENT
# ==================================

def gradient(X, theta, y):
    m = len(y)
    predictions = modele(X, theta)

    return (1 / m) * X.T @ (predictions - y)


# ==================================
# ETAPE 7 : DESCENTE DE GRADIENT
# ==================================

def descente_gradient(X, theta, alpha, y, n_iteration):

    historique = []

    for i in range(n_iteration):

        theta = theta - alpha * gradient(X, theta, y)

        historique.append(cout(X, theta, y))

    return theta, historique


# ==================================
# ETAPE 8 : APPRENTISSAGE
# ==================================

theta_initial = theta.copy()

new_theta, historique = descente_gradient(
    X, theta, 0.01, y, 1000
)

print("Theta final :")
print(new_theta)

print("Cout initial :", cout(X, theta_initial, y))
print("Cout final :", cout(X, new_theta, y))


# ==================================
# ETAPE 9 : VISUALISATION
# ==================================

plt.scatter(x, y, label="Données réelles")

plt.plot(
    x,
    modele(X, new_theta),
    color="red",
    label="Droite de régression"
)

plt.title("Régression linéaire")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


# ==================================
# ETAPE 10 : EVOLUTION DU COUT
# ==================================

plt.plot(historique)
plt.title("Evolution de la fonction de coût")
plt.xlabel("Itérations")
plt.ylabel("Coût")
plt.show()
