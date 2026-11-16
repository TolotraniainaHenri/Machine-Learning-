"""
Cours de Machine Learning : Régression linéaire (prix d'une maison)
Exercices corrigés avec NumPy
Résultat attendu : pour 120 m2, prix prédit = 60 M Ar
"""
import numpy as np

# ------------------------------------------------------------------
# EXERCICE 1 : Construire les matrices X et Y
# ------------------------------------------------------------------
print("=" * 60)
print("EXERCICE 1 : Construire X et Y")
print("=" * 60)

superficie = np.array([20, 40, 60, 80, 100], dtype=float)   # m2
prix = np.array([10, 20, 30, 40, 50], dtype=float)          # M Ar

m = len(superficie)                                         # nombre d'exemples
X = np.column_stack([np.ones(m), superficie])               # colonne de 1 (biais) + superficie
Y = prix.reshape(-1, 1)                                     # vecteur colonne (m x 1)

print("X =\n", X)
print("Y =\n", Y)
print("Forme de X :", X.shape, "| Forme de Y :", Y.shape)   # (5,2) et (5,1)

# ------------------------------------------------------------------
# EXERCICE 2 : Prédiction  Y_hat = X . theta
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("EXERCICE 2 : Prédiction avec theta = [0, 0.5]")
print("=" * 60)

theta = np.array([[0.0], [0.5]])      # [b ; a]
Y_hat = X @ theta
print("Y_hat =\n", Y_hat)
print("Le modèle prédit-il correctement ?", np.allclose(Y_hat, Y))

# ------------------------------------------------------------------
# EXERCICE 3 : Fonction de coût  J(theta) = 1/(2m) (X theta - Y)^T (X theta - Y)
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("EXERCICE 3 : Fonction de coût")
print("=" * 60)

def cout(X, Y, theta):
    m = len(Y)
    erreur = X @ theta - Y
    return ((erreur.T @ erreur) / (2 * m)).item()

theta_zero = np.zeros((2, 1))
print("J(theta = [0, 0])   =", cout(X, Y, theta_zero))   # 550.0
print("J(theta = [0, 0.5]) =", cout(X, Y, theta))        # 0.0

# ------------------------------------------------------------------
# EXERCICE 4 : Gradient  grad J = 1/m X^T (X theta - Y)  + une étape de mise à jour
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("EXERCICE 4 : Gradient et une étape")
print("=" * 60)

def gradient(X, Y, theta):
    m = len(Y)
    return (X.T @ (X @ theta - Y)) / m

g = gradient(X, Y, theta_zero)
print("Gradient en theta = [0, 0] :\n", g)               # [[-30], [-2200]]

# Vérification numérique (différences finies) : le gradient est-il correct ?
def gradient_numerique(X, Y, theta, eps=1e-5):
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        t_plus, t_moins = theta.copy(), theta.copy()
        t_plus[i] += eps
        t_moins[i] -= eps
        g[i] = (cout(X, Y, t_plus) - cout(X, Y, t_moins)) / (2 * eps)
    return g

print("Gradient numérique :\n", gradient_numerique(X, Y, theta_zero))

alpha = 0.0001
theta_nouveau = theta_zero - alpha * g
print("Après 1 étape (alpha = 0.0001) :\n", theta_nouveau)
print("Nouveau coût :", cout(X, Y, theta_nouveau))

# ------------------------------------------------------------------
# EXERCICE 5 : Descente de gradient complète
# Les superficies (20..100) sont grandes : on normalise pour converger vite,
# puis on revient aux unités d'origine.
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("EXERCICE 5 : Descente de gradient complète")
print("=" * 60)

mu, sigma = superficie.mean(), superficie.std()
X_n = np.column_stack([np.ones(m), (superficie - mu) / sigma])

def descente_gradient(X, Y, alpha, n_iter):
    theta = np.zeros((X.shape[1], 1))
    historique = []
    for i in range(n_iter):
        theta = theta - alpha * gradient(X, Y, theta)   # mise à jour
        historique.append(cout(X, Y, theta))
    return theta, historique

theta_n, historique = descente_gradient(X_n, Y, alpha=0.1, n_iter=200)

# Retour aux unités d'origine : a = a_n / sigma ; b = b_n - a_n * mu / sigma
a = theta_n[1, 0] / sigma
b = theta_n[0, 0] - theta_n[1, 0] * mu / sigma
print(f"theta trouvé : b = {b:.4f} , a = {a:.4f}   (attendu : b = 0 , a = 0.5)")
print(f"Coût initial = {historique[0]:.4f} | coût final = {historique[-1]:.2e}")

# ------------------------------------------------------------------
# EXERCICE 6 : Équation normale  theta = (X^T X)^(-1) X^T Y
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("EXERCICE 6 : Équation normale (solution exacte)")
print("=" * 60)

theta_exact = np.linalg.inv(X.T @ X) @ X.T @ Y
print("theta exact :\n", theta_exact.round(6))

# ------------------------------------------------------------------
# EXERCICE 7 : Prédire le prix d'une maison de 120 m2
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("EXERCICE 7 : Prédiction pour 120 m2")
print("=" * 60)

x_nouveau = np.array([[1.0, 120.0]])        # [1, superficie]
prix_predit = (x_nouveau @ np.array([[b], [a]])).item()
print(f"Prix prédit pour 120 m2 = {prix_predit:.2f} M Ar")   # 60.00

# ------------------------------------------------------------------
# BONUS : courbe d'erreur J en fonction des itérations (si matplotlib est installé)
# ------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(historique, "r-")
    plt.xlabel("Itérations")
    plt.ylabel("J(theta)")
    plt.title("L'erreur diminue progressivement")
    plt.grid(alpha=0.3)
    plt.savefig("courbe_cout.png", dpi=120, bbox_inches="tight")
    print("\nCourbe enregistrée : courbe_cout.png")
except ImportError:
    print("\n(matplotlib non installé : courbe ignorée)")
