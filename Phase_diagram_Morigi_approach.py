import numpy as np
from scipy.linalg import eigh 
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

z = 4
n_max = 23

epsilon = 1e-5

phi_e, phi_o, theta = 0.001, 0.002, 0.0

n = np.arange(0, n_max + 1)
n_diag = np.diag(n)

def hamiltonian_even(phi_e, phi_o, theta, mu_U0, zt_U0, U_inf_U0):
    a = np.diag(np.sqrt(np.arange(1, n_max + 1)), 1) 
    a_dag = a.T 
    
    identity_matrix = np.eye(n_max + 1)
    
    H_e = -zt_U0 * phi_o * (a + a_dag) + zt_U0 * phi_o * phi_e * identity_matrix \
        + 0.5 * (n_diag @ (n_diag - identity_matrix)) \
        - U_inf_U0 * theta * n_diag \
        + (U_inf_U0 / 4) * theta**2 * identity_matrix \
        - mu_U0 * n_diag
        
    return H_e

def hamiltonian_odd(phi_e, phi_o, theta, mu_U0, zt_U0, U_inf_U0):
    a = np.diag(np.sqrt(np.arange(1, n_max + 1)), 1) 
    a_dag = a.T 
    
    identity_matrix = np.eye(n_max + 1)
    
    H_o = -zt_U0 * phi_e * (a + a_dag) + zt_U0 * phi_e * phi_o * identity_matrix \
        + 0.5 * (n_diag @ (n_diag - identity_matrix)) \
        + U_inf_U0 * theta * n_diag \
        + (U_inf_U0 / 4) * theta**2 * identity_matrix \
        - mu_U0 * n_diag
        
    return H_o

def ground_state_expectation(hamiltonian):
    eigvals, eigvecs = eigh(hamiltonian) 
    ground_state = eigvecs[:, 0] 
    n_expect = np.vdot(ground_state, np.dot(np.diag(n), ground_state))  
    a_expect = np.vdot(ground_state, np.dot(np.diag(np.sqrt(np.arange(1, n_max + 1)), 1), ground_state))
    energy = np.vdot(ground_state, np.dot(hamiltonian, ground_state))  
    
    return a_expect, n_expect, energy

def fixed_point_iteration(mu_U0, zt_U0, U_inf_U0, phi_e, phi_o, theta, max_iters=300):
    for i in range(max_iters):
        H_e = hamiltonian_even(phi_e, phi_o, theta, mu_U0, zt_U0, U_inf_U0)
        H_o = hamiltonian_odd(phi_e, phi_o, theta, mu_U0, zt_U0, U_inf_U0)

        a_e, n_e, E_e = ground_state_expectation(H_e)
        a_o, n_o, E_o = ground_state_expectation(H_o)

        energy = 0.5 * (E_o + E_e)
        phi_e_new, phi_o_new = a_e, a_o
        theta_new = n_e - n_o

        delta_phi_e = np.abs(phi_e_new - phi_e)
        delta_phi_o = np.abs(phi_o_new - phi_o)
        delta_theta = np.abs(theta_new - theta)

        phi_e, phi_o, theta = phi_e_new, phi_o_new, theta_new
        
        if max(delta_phi_e, delta_phi_o, delta_theta) < epsilon:
            break
    density = (n_e + n_o) / 2


    return phi_e, phi_o, theta, density, energy


def plot_phase_diagram(zt_range, mu_range, U_inf_U0 = 0.6, resolution=80):
    zt_values = np.linspace(*zt_range, resolution)
    mu_values = np.linspace(*mu_range, resolution)
    varphi_matrix = np.zeros((resolution, resolution))
    theta_matrix = np.zeros((resolution, resolution))

    N_theta = 2
    initial_conditions = [(0, 0, n) for n in range(N_theta)] + \
                        [(0.001, 0.002, n) for n in range(N_theta)] + \
                        [(0.1, 0.2, n) for n in range(N_theta)]

    step = 0
    for i, zt_U0 in enumerate(zt_values):
        for j, mu_U0 in enumerate(mu_values):
            print('step',step + 1, '/', resolution**2)
            step += 1
            min_energy = np.inf
            best_varphi = 0
            best_theta = 0

            for initial in initial_conditions:
                phi_e, phi_o, theta = initial
                phi_e, phi_o, theta, density, energy = fixed_point_iteration(mu_U0, zt_U0, U_inf_U0, phi_e, phi_o, theta)

                if energy < min_energy:
                    min_energy = energy
                    best_varphi = np.sqrt(np.abs(phi_e * phi_o))
                    best_theta = abs(theta)

            varphi_matrix[j, i] = best_varphi
            theta_matrix[j, i] = best_theta


    fig, ax = plt.subplots(figsize=(8, 6))
    norm = Normalize(vmin=np.min(varphi_matrix), vmax=1.5)
    cbar = ScalarMappable(norm=norm, cmap='viridis')

    c = ax.pcolormesh(zt_values, mu_values, varphi_matrix, shading='auto', cmap='viridis', norm=norm)
    ax.set_xlabel('zt/U_0', fontsize=14)
    ax.set_ylabel('mu/U_0', fontsize=14)
    ax.set_title('varphi as a function of zt/U_0 and mu/U_0 with U_infty/U_0='+str(U_inf_U0), fontsize=16)
    fig.colorbar(c, ax=ax, label='varphi')
    plt.show()

    #theta
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = Normalize(vmin=np.min(theta_matrix), vmax=2)
    cbar = ScalarMappable(norm=norm, cmap='viridis')

    c = ax.pcolormesh(zt_values, mu_values, theta_matrix, shading='auto', cmap='viridis', norm=norm)
    ax.set_xlabel('zt/U_0', fontsize=14)
    ax.set_ylabel('mu/U_0', fontsize=14)
    ax.set_title('theta as a function of zt/U_0 and mu/U_0 with U_infty/U_0='+str(U_inf_U0), fontsize=16)
    fig.colorbar(c, ax=ax, label='|theta|')
    plt.show()


min_zt = float(input('min zt/U: '))
max_zt = float(input('max zt/U: '))
min_mu = float(input('min mu: '))
max_mu = float(input('max mu: '))
U_inf_U0 = float(input('value of U_cav/U: '))
zt_range=(min_zt, max_zt)
mu_range=(min_mu, max_mu)
resolution = int(input('resolution (suggested <80): '))
plot_phase_diagram(zt_range, mu_range, U_inf_U0, resolution)
