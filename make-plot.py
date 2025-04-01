import numpy as np
import matplotlib.pyplot as plt

# Leer el archivo y filtrar las líneas de metadatos (comentarios)
with open("results.txt", "r") as f:
    lines = [line for line in f if not line.startswith("#")]

# Usamos np.genfromtxt sobre las líneas filtradas
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
O_B      = data["O_B"]
O_DW     = data["O_DW"]

# Graficar todas las variables
plt.figure(figsize=(8,6))
plt.semilogx(t_over_U, n1, marker='', label="n1")
plt.semilogx(t_over_U, var_n1, marker='', label="var_n1")
plt.semilogx(t_over_U, n2, marker='', label="n2")
plt.semilogx(t_over_U, var_n2, marker='', label="var_n2")
plt.semilogx(t_over_U, sf_factor, marker='', label="sf_factor")
plt.semilogx(t_over_U, S_half, marker='', label="S_half")
plt.semilogx(t_over_U, S_even, marker='', label="S_even")
plt.semilogx(t_over_U, O_B, marker='', label="O_B")
plt.semilogx(t_over_U, O_DW, marker='', label="O_DW")

plt.xlabel("t/U")
plt.ylabel("Valor")
plt.title("Resultados vs t/U")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
