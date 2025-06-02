import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Parámetros globales
Ns    = 100
N_eff = Ns / 2.0   # Factor de corrección usado en mu
z     = 6
J_D   = 1.0
U     = 1.0
g_eff = -0.005
n_max = 31
d= int(z/2)

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

def create_beta_op(site, psi_local, psi_neighbor):
    return  psi_neighbor * (a_op + adag_op - psi_local * Id)

def build_hamiltonian(site, psi_neighbor, psi_local, mu_local, U_local, rho_local, C_D, zt0, g_eff, beta):
    term_hopping = beta
    term_mu      = - mu_local * n_op
    term_U       = (U_local / 2.0) * n_n_minus_one
    term_g       = - g_eff * J_D**2 * rho_local * n_op
    term_const   = - g_eff * C_D * Id
    H_local = - zt0 * term_hopping + term_mu + term_U + term_g + term_const
    return H_local * Ns/2.0

def fixed_point_iteration(mu, zt0, psi_o, psi_e, rho_o, rho_e, max_iters=300):
    for i in range(max_iters):
        Delta_rho = (rho_o - rho_e) / 2.0
        
        mu_e = mu - 2 * g_eff * N_eff * J_D**2 * Delta_rho
        mu_o = mu + 2 * g_eff * N_eff * J_D**2 * Delta_rho

        U_e = U + 2 * g_eff * J_D**2
        U_o = U + 2 * g_eff * J_D**2 

        C_D_e = (- N_eff*2 * J_D**2 * Delta_rho * rho_e) / 2.0 - (J_D**2*rho_e**2) / 2.0
        C_D_o = (  N_eff*2 * J_D**2 * Delta_rho * rho_o) / 2.0 - (J_D**2*rho_o**2) / 2.0
        
        beta_e = create_beta_op('e', psi_e, psi_o)
        beta_o = create_beta_op('o', psi_o, psi_e)
        beta   = 0.5 * (beta_e + beta_o)
        
        H_e = build_hamiltonian('e', psi_o, psi_e, mu_e, U_e, rho_e, C_D_e, zt0, g_eff, beta)
        H_o = build_hamiltonian('o', psi_e, psi_o, mu_o, U_o, rho_o, C_D_o, zt0, g_eff, beta)
        

        eigvals_e, eigvecs_e = eigh(H_e)
        eigvals_o, eigvecs_o = eigh(H_o)
        gs_e = eigvecs_e[:, 0]
        gs_o = eigvecs_o[:, 0]
        
        new_psi_e = np.vdot(gs_e, np.dot(a_op, gs_e))
        new_psi_o = np.vdot(gs_o, np.dot(a_op, gs_o))
        new_rho_e = np.vdot(gs_e, np.dot(n_op, gs_e)).real
        new_rho_o = np.vdot(gs_o, np.dot(n_op, gs_o)).real
        energy    = np.vdot(gs_e, np.dot(H_e, gs_e)) + np.vdot(gs_o, np.dot(H_o, gs_o))
        
        if max(abs(new_psi_e - psi_e), abs(new_psi_o - psi_o),
               abs(new_rho_e - rho_e), abs(new_rho_o - rho_o)) < 1e-5:
            psi_e, psi_o, rho_e, rho_o = new_psi_e, new_psi_o, new_rho_e, new_rho_o
            break

        psi_e, psi_o = new_psi_e, new_psi_o
        rho_e, rho_o = new_rho_e, new_rho_o

    density = (rho_e + rho_o) / 2.0
    return psi_o, psi_e, rho_o, rho_e, density, energy, eigvecs_o, eigvecs_e

def plot_phase_diagram(zt_range, mu_range, resolution, N_cond=12):
    zt_values = np.linspace(*zt_range, resolution)
    mu_values = np.linspace(*mu_range, resolution)
    total_psi_matrix = np.zeros((resolution, resolution))
    imbalance_matrix = np.zeros((resolution, resolution)) 

    for i, zt0 in enumerate(zt_values):
        for j, mu in enumerate(mu_values):
            min_energy = np.inf
            best_total_psi = 0
            best_imbalance = 0

            # Se prueban varias condiciones iniciales para buscar el mínimo de energía
            for k in range(N_cond):
                psi_e0 = np.random.uniform(0, 0.3)
                psi_o0 = np.random.uniform(0, 0.3)
                rho_e0 = np.random.uniform(0, 1.4)
                rho_o0 = np.random.uniform(0, 1.4)
                
                psi_o, psi_e, rho_o, rho_e, density, energy, ground_state_odd, ground_state_even = fixed_point_iteration(mu, zt0, psi_o0, psi_e0, rho_o0, rho_e0)
                
                if energy < min_energy:
                    min_energy = energy
                    total_psi = 0.5 * (abs(psi_e) + abs(psi_o))
                    imbalance = 0.5 * (rho_o - rho_e)
                    best_total_psi = total_psi
                    best_imbalance = abs(imbalance)
                    best_psi_o, best_psi_e = psi_o, psi_e
                    best_rho_o, best_rho_e = rho_o, rho_e

            print(f"({zt0:.3f},{mu:.3f}): (po, pe, ro, re) = ({best_psi_o:.3f}, {best_psi_e:.3f}, {best_rho_o:.3f}, {best_rho_e:.3f}) => im = {best_imbalance:.2f}, sf = {best_total_psi:.2f}")

            total_psi_matrix[j, i] = best_total_psi
            imbalance_matrix[j, i] = best_imbalance

    # Gráfica de total_psi
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = Normalize(vmin=np.min(total_psi_matrix), vmax=np.max(total_psi_matrix))
    cbar = ScalarMappable(norm=norm, cmap='viridis')
    c = ax.pcolormesh(zt_values, mu_values, total_psi_matrix, shading='auto', norm=norm)
    ax.set_xlabel('zt0', fontsize=14)
    ax.set_ylabel('mu', fontsize=14)
    ax.set_title('total_psi', fontsize=16)
    fig.colorbar(c, ax=ax, label='total_psi')
    plt.show()

    # Gráfica de imbalance
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = Normalize(vmin=np.min(imbalance_matrix), vmax=np.max(imbalance_matrix))
    cbar = ScalarMappable(norm=norm, cmap='viridis')
    c = ax.pcolormesh(zt_values, mu_values, imbalance_matrix, shading='auto', norm=norm)
    ax.set_xlabel('zt0', fontsize=14)
    ax.set_ylabel('mu', fontsize=14)
    ax.set_title('Imbalance', fontsize=16)
    fig.colorbar(c, ax=ax, label='Imbalance')
    plt.show()

def one_point(zt0, mu, N_cond=12):
    
    min_energy = np.inf
    best_total_psi = 0
    best_imbalance = 0

    for k in range(N_cond):
        psi_e0 = np.random.uniform(0, 0.3)
        psi_o0 = np.random.uniform(0, 0.3)
        rho_e0 = np.random.uniform(0, 1.4)
        rho_o0 = np.random.uniform(0, 1.4)
        
        psi_o, psi_e, rho_o, rho_e, density, energy, eigvecs_o, eigvecs_e = fixed_point_iteration(mu, zt0, psi_o0, psi_e0, rho_o0, rho_e0)
        
        if energy < min_energy:
            min_energy = energy
            total_psi = 0.5 * (abs(psi_e) + abs(psi_o))
            imbalance = 0.5 * (rho_o - rho_e)
            best_total_psi = total_psi
            best_imbalance = abs(imbalance)
            best_psi_o, best_psi_e = psi_o, psi_e
            best_rho_o, best_rho_e = rho_o, rho_e
            best_eigvecs_o, best_eigvecs_e = eigvecs_o, eigvecs_e
    return best_total_psi, best_imbalance, best_psi_o, best_psi_e, best_rho_o, best_rho_e,  best_eigvecs_o, best_eigvecs_e

def multipoints():
    min_zt = 0.0
    max_zt = 0.2
    min_mu = 0
    max_mu = 2.5
    zt_range = (min_zt, max_zt)
    mu_range = (min_mu, max_mu)
    resolution = 20
    plot_phase_diagram(zt_range, mu_range, resolution)

multipoints()
#total_psi, imbalance, psi_o, psi_e, rho_o, rho_e, eigvecs_o, eigvecs_e =one_point(.1, .42)
