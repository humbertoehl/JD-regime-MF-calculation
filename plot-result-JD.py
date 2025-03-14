import numpy as np
import matplotlib.pyplot as plt

def load_phase_data(filename):
    data = np.genfromtxt(filename, comments='#', delimiter='\t')
    return data

def plot_phase(param, filename="phase_diagram_results.txt"):
    param_dict = {
        "mu": 0,
        "zt/U": 1,
        "psi_o": 2,
        "psi_e": 3,
        "rho_o": 4,
        "rho_e": 5,
        "psi_total": 6,
        "delta_rho": 7
    }
    
    if param not in param_dict:
        raise ValueError(f"El parámetro '{param}' no está disponible. Opciones: {list(param_dict.keys())}")
    
    # Cargar datos
    data = load_phase_data(filename)
    
    mu_vals = np.unique(data[:, 0])
    zt_vals = np.unique(data[:, 1])
    n_mu = len(mu_vals)
    n_zt = len(zt_vals)
    
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

    input_filename = "./Resultados/JD-Results(8).txt"

    plot_phase("psi_total", filename=input_filename)
    plot_phase("delta_rho", filename=input_filename)
