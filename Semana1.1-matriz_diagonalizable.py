"""
PROBLEMA 3 - MODELO DE LEONTIEF DEPENDIENTE DEL TIEMPO
Resolución en Python usando:
1) Diagonalización
2) Matriz exponencial directa
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# ------------------------------------------------------------
# 1. Definición del Problema
# ------------------------------------------------------------
# Sistema diferencial:
#    dX/dt = A X
# con condición inicial:
#    X(0) = X0
#
# Donde:
#    A  = matriz de insumo-producto
#    X0 = vector de producción inicial
# ------------------------------------------------------------

A = np.array([[0.5, 0.2],
              [0.1, 0.4]], dtype=float)

X0 = np.array([[120],
               [70]], dtype=float)

# Intervalo de tiempo solicitado: [0, 20]
t_values = np.linspace(0, 20, 200)

# ------------------------------------------------------------
# 2. MÉTODO 1: Resulución mediante diagonalización
# ------------------------------------------------------------
# Si A es diagonalizable, entonces:
#    A = P D P^{-1}
#
# y por tanto:
#    e^(At) = P e^(Dt) P^{-1}
#
# Entonces la solución del sistema es:
#    X(t) = e^(At) X0 = P e^(Dt) P^{-1} X0
# ------------------------------------------------------------

# Obtenemos autovalores y autovectores
eigenvalues, P = np.linalg.eig(A)

# Matriz diagonal D formada por los autovalores
D = np.diag(eigenvalues)

# Inversa de la matriz de autovectores
P_inv = np.linalg.inv(P)

# Función que calcula X(t) por diagonalización
def X_diagonalizacion(t):
    """
    Calcula la solución X(t) usando diagonalización:
        X(t) = P * exp(Dt) * P^{-1} * X0
    """
    exp_Dt = np.diag(np.exp(eigenvalues * t))
    Xt = P @ exp_Dt @ P_inv @ X0
    return Xt

# ------------------------------------------------------------
# 3. MÉTODO 2: Resolución mediante matriz exponencial directa
# ------------------------------------------------------------
# La solución general del sistema es:
#    X(t) = e^(At) X0
#
# Aquí usamos scipy.linalg.expm para calcular directamente
# la exponencial matricial e^(At).
# ------------------------------------------------------------

def X_exponencial(t):
    """
    Calcula la solución X(t) usando la matriz exponencial:
        X(t) = expm(A*t) * X0
    """
    Xt = expm(A * t) @ X0
    return Xt

# ------------------------------------------------------------
# 4. Evaluar ambos métodos en el intervalo [0, 20]
# ------------------------------------------------------------
# Vamos a calcular ambas soluciones para muchos valores de t
# y verificar que coincidan numéricamente.
# ------------------------------------------------------------

sol_diag = []
sol_exp = []

for t in t_values:
    sol_diag.append(X_diagonalizacion(t).flatten())
    sol_exp.append(X_exponencial(t).flatten())

sol_diag = np.array(sol_diag)
sol_exp = np.array(sol_exp)

# ------------------------------------------------------------
# 5. Verificación numérica
# ------------------------------------------------------------
# Comparamos ambas soluciones mediante:
# - error absoluto máximo
# - error relativo máximo
# - comparación con tolerancia numérica
# ------------------------------------------------------------

error_maximo = np.max(np.abs(sol_diag - sol_exp))
error_relativo = np.max(
    np.abs(sol_diag - sol_exp) / np.maximum(np.abs(sol_exp), 1e-15)
)

coinciden = np.allclose(sol_diag, sol_exp, rtol=1e-10, atol=1e-8)

print("===================================================")
print("AUTOVALORES:")
print(eigenvalues)
print("===================================================")
print("AUTOVECTORES (columnas de P):")
print(P)
print("===================================================")
print("MATRIZ DIAGONAL D:")
print(D)
print("===================================================")
print("INVERSA DE P:")
print(P_inv)
print("===================================================")
print(f"Error absoluto máximo: {error_maximo:.12e}")
print(f"Error relativo máximo: {error_relativo:.12e}")
print("===================================================")

# ------------------------------------------------------------
# 6. Graficación de las soluciones
# ------------------------------------------------------------
# Graficamos las componentes x1(t) y x2(t) usando uno de los
# métodos (como ambos coinciden, basta con uno).
# ------------------------------------------------------------

x1 = sol_exp[:, 0]
x2 = sol_exp[:, 1]

plt.figure(figsize=(10, 6))
plt.plot(t_values, x1, label='x1(t): Producción sector 1')
plt.plot(t_values, x2, label='x2(t): Producción sector 2')
plt.xlabel('Tiempo t')
plt.ylabel('Producción')
plt.title('Modelo de Leontief dependiente del tiempo')
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# 7. Conclusión
# ------------------------------------------------------------

if coinciden:
    print("\nConclusión:")
    print("Ambos métodos producen prácticamente la misma solución numérica.")
    print("Las pequeñas diferencias observadas se deben al redondeo numérico.")
    print("Por tanto, la diagonalización y la matriz exponencial son equivalentes en este problema.")
else:
    print("\nConclusión:")
    print("Las soluciones no coinciden dentro de la tolerancia establecida.")
    print("Se recomienda revisar la implementación.")
