import numpy as np
import matplotlib.pyplot as plt

# Leer el archivo y filtrar las líneas de metadatos (comentarios)
with open("results.txt", "r") as f:
    lines = [line for line in f if not line.startswith("#")]

# Ahora usamos np.genfromtxt sobre las líneas filtradas
data = np.genfromtxt(lines, delimiter="\t", names=True)

# Extraer las columnas
t_over_U = data["t_U"]
n1       = data["n1"]
var_n1   = data["var_n1"]
n2       = data["n2"]
var_n2   = data["var_n2"]
sf_factor= data["sf_factor"]
S_half   = data["S_half"]
S_even   = data["S_even"]

plt.figure(figsize=(8,6))
plt.semilogx(t_over_U, n1, marker='o', label="n1")
plt.semilogx(t_over_U, var_n1, marker='s', label="var_n1")
plt.semilogx(t_over_U, n2, marker='^', label="n2")
plt.semilogx(t_over_U, var_n2, marker='v', label="var_n2")
plt.semilogx(t_over_U, sf_factor, marker='d', label="sf_factor")
plt.semilogx(t_over_U, S_half, marker='*', label="S_half")
plt.semilogx(t_over_U, S_even, marker='x', label="S_even")

plt.xlabel("t/U")
plt.ylabel("Valor")
plt.title("Resultados vs t/U")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
