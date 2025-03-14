import numpy as np
import matplotlib.pyplot as plt

def load_phase_data(filename):
    data = np.genfromtxt(filename, comments='#', delimiter='\t')
    return data

def plot_phase(param, filename="JB-Results(12).txt"):
    # Diccionario de parámetros: índices según columnas del archivo.
    param_dict = {
        "mu": 0,
        "zt/U": 1,
        "psi1": 2,
        "psi2": 3,
        "psi3": 4,
        "psi4": 5,
        "rho1": 6,
        "rho2": 7,
        "rho3": 8,
        "rho4": 9,
        "|psi|_avg": 10,
        "avg_phase_diff": 11,
        "phase1": 12,
        "phase2": 13,
        "phase3": 14,
        "phase4": 15
    }
    
    if param not in param_dict:
        raise ValueError(f"El parámetro '{param}' no está disponible. Opciones: {list(param_dict.keys())}")
    
    # Cargar los datos del archivo
    data = load_phase_data(filename)
    
    # Suponemos que el barrido se realizó en dos variables: mu y zt/U.
    mu_vals = np.unique(data[:, 0])
    zt_vals = np.unique(data[:, 1])
    n_mu = len(mu_vals)
    n_zt = len(zt_vals)
    
    # Extraer el parámetro solicitado y darle forma de grilla
    param_index = param_dict[param]
    param_data = data[:, param_index].reshape(n_mu, n_zt)
    
    extent = [zt_vals[0], zt_vals[-1], mu_vals[0], mu_vals[-1]]
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(param_data, interpolation='nearest', origin='lower', extent=extent, aspect='auto', cmap='viridis')
    plt.colorbar(im, label=param)
    plt.xlabel("zt/U")
    plt.ylabel("mu/U")
    plt.title(f"Mapa de {param}")
    plt.show()

if __name__ == '__main__':
    # Ajusta el nombre del archivo si es necesario
    input_filename = "./Resultados/JB-Results(7).txt"
    
    # Ejemplos de gráficas: campo superfluido promedio y diferencia de fases
    plot_phase("phase2", filename=input_filename)
    plot_phase("phase1", filename=input_filename)
