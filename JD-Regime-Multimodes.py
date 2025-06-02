import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Parámetros globales
Ns    = 100
z     = 6
J_D   = 1.0
U     = 1.0
g_eff = -0.005
n_max = 6

# Definición de operadores
def create_annihilation(n_max):
    a = np.zeros((n_max, n_max))
    for n in range(1, n_max):
        a[n-1, n] = np.sqrt(n)
    return a

def create_creation(n_max):
    return create_annihilation(n_max).T

def create_number(n_max):
    return np.diag(np.arange(n_max))

a_op    = create_annihilation(n_max)
adag_op = create_creation(n_max)
n_op    = create_number(n_max)
Id      = np.eye(n_max)
n_n_minus_one = np.zeros((n_max, n_max))
for n in range(n_max):
    n_n_minus_one[n, n] = n * (n - 1)

# Operador beta para el sistema de 3 sublattices
def create_beta_op_three(psi_m, psi, psi_p):
    # psi_m: psi de la sublattice "anterior" (ξ-1)
    # psi_p: psi de la sublattice "siguiente" (ξ+1)
    term1 = (np.conjugate(psi_p) + np.conjugate(psi_m)) * a_op
    term2 = (psi_p + psi_m) * adag_op
    constant = 0.5 * (np.conjugate(psi) * (psi_p + psi_m) + psi * (np.conjugate(psi_p) + np.conjugate(psi_m)))
    return term1 + term2 - constant * Id

# Construcción del Hamiltoniano efectivo para cada sublattice
def build_hamiltonian_three(psi_m, psi, psi_p, mu_local, U_local, zt0):
    beta = create_beta_op_three(psi_m, psi, psi_p)/2
    term_mu = - mu_local * n_op
    term_U  = (U_local/2.0) * n_n_minus_one
    H_local = zt0 * beta + term_mu + term_U
    return H_local * (Ns/3.0)

# Iteración de punto fijo para el sistema de 3 sublattices
def fixed_point_iteration_three(mu, zt0, psi1, psi2, psi3, rho1, rho2, rho3, max_iters=300):
    for i in range(max_iters):
        # Actualización de parámetros efectivos para cada sublattice
        mu1 = mu - (g_eff * Ns * J_D**2 / 3.0) * (rho1 - (rho2 + rho3)/2.0)
        mu2 = mu - (g_eff * Ns * J_D**2 / 3.0) * (rho2 - (rho3 + rho1)/2.0)
        mu3 = mu - (g_eff * Ns * J_D**2 / 3.0) * (rho3 - (rho1 + rho2)/2.0)
        U_eff = U + 2 * g_eff * J_D**2
        
        # Construcción de Hamiltonianos para cada sublattice
        # Se utiliza la condición periódica: para ξ=1, el vecino “izquierdo” es ξ=3; para ξ=3, el vecino “derecho” es ξ=1.
        H1 = build_hamiltonian_three(psi3, psi1, psi2, mu1, U_eff, zt0)
        H2 = build_hamiltonian_three(psi1, psi2, psi3, mu2, U_eff, zt0)
        H3 = build_hamiltonian_three(psi2, psi3, psi1, mu3, U_eff, zt0)
        
        # Diagonalización y obtención del estado fundamental
        eigvals1, eigvecs1 = eigh(H1)
        eigvals2, eigvecs2 = eigh(H2)
        eigvals3, eigvecs3 = eigh(H3)
        
        gs1 = eigvecs1[:, 0]
        gs2 = eigvecs2[:, 0]
        gs3 = eigvecs3[:, 0]
        
        # Actualización de los parámetros de orden y densidad
        new_psi1 = np.vdot(gs1, np.dot(a_op, gs1))
        new_psi2 = np.vdot(gs2, np.dot(a_op, gs2))
        new_psi3 = np.vdot(gs3, np.dot(a_op, gs3))
        
        new_rho1 = np.vdot(gs1, np.dot(n_op, gs1)).real
        new_rho2 = np.vdot(gs2, np.dot(n_op, gs2)).real
        new_rho3 = np.vdot(gs3, np.dot(n_op, gs3)).real
        
        # Energía total
        energy = (np.vdot(gs1, np.dot(H1, gs1)) +
                  np.vdot(gs2, np.dot(H2, gs2)) +
                  np.vdot(gs3, np.dot(H3, gs3)))
        
        # Verificación de convergencia
        diff = max(abs(new_psi1 - psi1), abs(new_psi2 - psi2), abs(new_psi3 - psi3),
                   abs(new_rho1 - rho1), abs(new_rho2 - rho2), abs(new_rho3 - rho3))
        if diff < 1e-5:
            psi1, psi2, psi3 = new_psi1, new_psi2, new_psi3
            rho1, rho2, rho3 = new_rho1, new_rho2, new_rho3
            break
        
        psi1, psi2, psi3 = new_psi1, new_psi2, new_psi3
        rho1, rho2, rho3 = new_rho1, new_rho2, new_rho3
        
    avg_density = (rho1 + rho2 + rho3) / 3.0
    return psi1, psi2, psi3, rho1, rho2, rho3, avg_density, energy, eigvecs1, eigvecs2, eigvecs3

# Función para obtener la solución en un punto (zt0, mu) probando N_cond condiciones iniciales
def one_point_three(zt0, mu, N_cond=12):
    min_energy = np.inf
    best_avg_psi = 0
    best_psi1 = best_psi2 = best_psi3 = 0
    best_rho1 = best_rho2 = best_rho3 = 0
    best_avg_rho = 0
    
    for k in range(N_cond):
        psi10 = np.random.uniform(0, 0.3)
        psi20 = np.random.uniform(0, 0.3)
        psi30 = np.random.uniform(0, 0.3)
        rho10 = np.random.uniform(0, 1.4)
        rho20 = np.random.uniform(0, 1.4)
        rho30 = np.random.uniform(0, 1.4)
        
        psi1, psi2, psi3, rho1, rho2, rho3, avg_density, energy, ev1, ev2, ev3 = fixed_point_iteration_three(
            mu, zt0, psi10, psi20, psi30, rho10, rho20, rho30)
        
        if energy < min_energy:
            min_energy = energy
            avg_psi = (abs(psi1) + abs(psi2) + abs(psi3)) / 3.0
            best_avg_psi = avg_psi
            best_psi1, best_psi2, best_psi3 = psi1, psi2, psi3
            best_rho1, best_rho2, best_rho3 = rho1, rho2, rho3
            best_avg_rho = avg_density
            
    print(f"zt0 = {zt0:.3f}, mu = {mu:.3f}: psi = ({best_psi1:.3f}, {best_psi2:.3f}, {best_psi3:.3f}), "
          f"rho = ({best_rho1:.3f}, {best_rho2:.3f}, {best_rho3:.3f}), avg(|psi|) = {best_avg_psi:.3f}, avg(rho) = {best_avg_rho:.3f}")
    return best_avg_psi, best_rho1, best_rho2, best_rho3, best_avg_rho

# Función para graficar el diagrama de fase: se muestran avg(|psi|) y las densidades rho_1, rho_2, rho_3, y avg(rho)
def plot_phase_diagram_three(zt_range, mu_range, resolution, N_cond=12):
    zt_values = np.linspace(*zt_range, resolution)
    mu_values = np.linspace(*mu_range, resolution)
    
    avg_psi_matrix = np.zeros((resolution, resolution))
    rho1_matrix = np.zeros((resolution, resolution))
    rho2_matrix = np.zeros((resolution, resolution))
    rho3_matrix = np.zeros((resolution, resolution))
    avg_rho_matrix = np.zeros((resolution, resolution))
    
    for i, zt0 in enumerate(zt_values):
        for j, mu in enumerate(mu_values):
            avg_psi, rho1, rho2, rho3, avg_rho = one_point_three(zt0, mu, N_cond)
            avg_psi_matrix[j, i] = avg_psi
            rho1_matrix[j, i] = rho1
            rho2_matrix[j, i] = rho2
            rho3_matrix[j, i] = rho3
            avg_rho_matrix[j, i] = avg_rho
    
    # Gráfica de avg(|psi|)
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = Normalize(vmin=np.min(avg_psi_matrix), vmax=np.max(avg_psi_matrix))
    c = ax.pcolormesh(zt_values, mu_values, avg_psi_matrix, shading='auto', norm=norm, cmap='viridis')
    ax.set_xlabel('zt0', fontsize=14)
    ax.set_ylabel('mu', fontsize=14)
    ax.set_title('avg(|psi|)', fontsize=16)
    fig.colorbar(c, ax=ax, label='avg(|psi|)')
    plt.show()
    
    # Gráficas de densidades para cada sublattice y avg(rho) en una grilla 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()  # Aplanar la grilla para iterar fácilmente
    titles = ['rho_1', 'rho_2', 'rho_3', 'avg(rho)']
    matrices = [rho1_matrix, rho2_matrix, rho3_matrix, avg_rho_matrix]
    for ax, mat, title in zip(axes, matrices, titles):
        norm = Normalize(vmin=np.min(mat), vmax=np.max(mat))
        c = ax.pcolormesh(zt_values, mu_values, mat, shading='auto', norm=norm, cmap='viridis')
        ax.set_xlabel('zt0', fontsize=14)
        ax.set_ylabel('mu', fontsize=14)
        ax.set_title(title, fontsize=16)
        fig.colorbar(c, ax=ax, label=title)
    plt.tight_layout()
    plt.show()

# Función multipoints que llama a la rutina de graficación
def multipoints_three():
    min_zt = 0
    max_zt = 0.165
    min_mu = -.12
    max_mu = .2
    zt_range = (min_zt, max_zt)
    mu_range = (min_mu, max_mu)
    resolution = 20
    plot_phase_diagram_three(zt_range, mu_range, resolution)

# Llamada para generar el diagrama de fase en el sistema de 3 sublattices
multipoints_three()
