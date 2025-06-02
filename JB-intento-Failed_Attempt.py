import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Parámetros globales
Ns    = 100
z     = 6
J_B   = 0.5
U     = 1.0
g_eff = -0.25
n_max = 6

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

n_n_minus_one = np.zeros((n_max, n_max))
for n in range(n_max):
    n_n_minus_one[n, n] = n * (n - 1)


def create_beta_phi(b_local, psi_local, psi_right):

    Id = np.eye(b_local.shape[0])
    # hopping
    hop1 = np.conj(psi_local) * psi_right * Id
    hop2 = np.conj(psi_right) * b_local
    hop3 = psi_right      * b_local.conj().T
    hop4 = psi_local      * np.conj(psi_right) * Id

    # constante
    overlap = np.conj(psi_local) * psi_right
    const  = (overlap + np.conj(overlap)) * Id

    return hop1 + hop2 + hop3 + hop4 - const


def build_hamiltonian_phi(mu_local, U_local, delta_S2_phi, ctilde_B_phi, t_phi, beta_op, n_op, n_n_minus_one, J_B, g_eff, Ns, z):

    Id = np.eye(n_op.shape[0])

    term_hop   = (z) * t_phi * beta_op
    term_mu    = - mu_local * n_op
    term_U     = (U_local/2) * n_n_minus_one
    term_delta =   g_eff * (J_B**2) * delta_S2_phi
    term_const = - g_eff * Ns * (J_B**2) * ctilde_B_phi * Id

    H_loc = term_hop + term_mu + term_U + term_delta + term_const
    return (Ns/4.0) * H_loc


def fixed_point_iteration(mu, t0, psi_1, psi_2, psi_3, psi_4, rho_1, rho_2, rho_3, rho_4, max_iters=300):
    Id = np.eye(n_op.shape[0])
    for i in range(max_iters):
      
        # Los dejo explícitamente por sitio para verificar que estén bien construídos
        sum_eta = (  (np.conj(psi_1)*psi_2 + np.conj(psi_2)*psi_1) 
                   - (np.conj(psi_2)*psi_3 + np.conj(psi_3)*psi_2) 
                   + (np.conj(psi_3)*psi_4 + np.conj(psi_4)*psi_3) 
                   - (np.conj(psi_4)*psi_1 + np.conj(psi_1)*psi_4))

        eta_1 = (z/8)*sum_eta
        eta_2 = -(z/8)*sum_eta
        eta_3 = (z/8)*sum_eta
        eta_4 = -(z/8)*sum_eta


        t_phi_1 = t0 - g_eff*Ns*J_B**2*eta_1
        t_phi_2 = t0 - g_eff*Ns*J_B**2*eta_2
        t_phi_3 = t0 - g_eff*Ns*J_B**2*eta_3
        t_phi_4 = t0 - g_eff*Ns*J_B**2*eta_4

        C_B_1 = (z/4) * (np.conj(psi_1)*psi_2 + np.conj(psi_2)*psi_1)*eta_1
        C_B_2 = (z/4) * (np.conj(psi_2)*psi_3 + np.conj(psi_3)*psi_2)*eta_2
        C_B_3 = (z/4) * (np.conj(psi_3)*psi_4 + np.conj(psi_4)*psi_3)*eta_3
        C_B_4 = (z/4) * (np.conj(psi_4)*psi_1 + np.conj(psi_1)*psi_4)*eta_4

        beta_1 = create_beta_phi(a_op, psi_1, psi_2)
        beta_2 = create_beta_phi(a_op, psi_2, psi_3)
        beta_3 = create_beta_phi(a_op, psi_3, psi_4)
        beta_4 = create_beta_phi(a_op, psi_4, psi_1)
        
        b_op = a_op
        b_dag = a_op.conj().T

        delta_S2_1 = (z/4) * ((b_dag @ b_dag)*psi_2**2 + np.conj(psi_2**2)*(b_op @ b_op) + 2*n_op*np.conj(psi_2)*psi_2  + n_op + np.conj(psi_2)*psi_2 * Id + 
                              (b_dag @ b_dag)*psi_4**2 + np.conj(psi_4**2)*(b_op @ b_op) + 2*n_op*np.conj(psi_4)*psi_4  + n_op + np.conj(psi_4)*psi_4 * Id -
                              2*(np.conj(psi_1)*psi_2 + np.conj(psi_2)*psi_1)*beta_1 + (np.conj(psi_1)*psi_2 + np.conj(psi_2)*psi_1)**2 * Id -
                              2*(np.conj(psi_1)*psi_4 + np.conj(psi_4)*psi_1)*beta_1 + (np.conj(psi_1)*psi_4 + np.conj(psi_4)*psi_1)**2 * Id)
        
        delta_S2_2 = (z/4) * ((b_dag @ b_dag)*psi_3**2 + np.conj(psi_3**2)*(b_op @ b_op) + 2*n_op*np.conj(psi_3)*psi_3  + n_op + np.conj(psi_3)*psi_3 * Id + 
                              (b_dag @ b_dag)*psi_1**2 + np.conj(psi_1**2)*(b_op @ b_op) + 2*n_op*np.conj(psi_1)*psi_1  + n_op + np.conj(psi_1)*psi_1 * Id -
                              2*(np.conj(psi_2)*psi_3 + np.conj(psi_3)*psi_2)*beta_2 + (np.conj(psi_2)*psi_3 + np.conj(psi_3)*psi_2)**2 * Id -
                              2*(np.conj(psi_2)*psi_1 + np.conj(psi_1)*psi_2)*beta_2 + (np.conj(psi_2)*psi_1 + np.conj(psi_1)*psi_2)**2 * Id)
        
        delta_S2_3 = (z/4) * ((b_dag @ b_dag)*psi_4**2 + np.conj(psi_4**2)*(b_op @ b_op) + 2*n_op*np.conj(psi_4)*psi_4  + n_op + np.conj(psi_4)*psi_4 * Id + 
                              (b_dag @ b_dag)*psi_2**2 + np.conj(psi_2**2)*(b_op @ b_op) + 2*n_op*np.conj(psi_2)*psi_2  + n_op + np.conj(psi_2)*psi_2 * Id -
                              2*(np.conj(psi_3)*psi_4 + np.conj(psi_4)*psi_3)*beta_3 + (np.conj(psi_3)*psi_4 + np.conj(psi_4)*psi_3)**2 * Id -
                              2*(np.conj(psi_3)*psi_2 + np.conj(psi_2)*psi_3)*beta_3 + (np.conj(psi_3)*psi_2 + np.conj(psi_2)*psi_3)**2 * Id)
        
        delta_S2_4 = (z/4) * ((b_dag @ b_dag)*psi_1**2 + np.conj(psi_1**2)*(b_op @ b_op) + 2*n_op*np.conj(psi_1)*psi_1  + n_op + np.conj(psi_1)*psi_1 * Id + 
                              (b_dag @ b_dag)*psi_3**2 + np.conj(psi_3**2)*(b_op @ b_op) + 2*n_op*np.conj(psi_3)*psi_3  + n_op + np.conj(psi_3)*psi_3 * Id -
                              2*(np.conj(psi_4)*psi_1 + np.conj(psi_1)*psi_4)*beta_4 + (np.conj(psi_4)*psi_1 + np.conj(psi_1)*psi_4)**2 * Id -
                              2*(np.conj(psi_4)*psi_3 + np.conj(psi_3)*psi_4)*beta_4 + (np.conj(psi_4)*psi_3 + np.conj(psi_3)*psi_4)**2 * Id)


        H_1 = build_hamiltonian_phi(mu, U, delta_S2_1, C_B_1, t_phi_1, beta_1, n_op, n_n_minus_one, J_B, g_eff, Ns, z)
        H_2 = build_hamiltonian_phi(mu, U, delta_S2_2, C_B_2, t_phi_2, beta_2, n_op, n_n_minus_one, J_B, g_eff, Ns, z)
        H_3 = build_hamiltonian_phi(mu, U, delta_S2_3, C_B_3, t_phi_3, beta_3, n_op, n_n_minus_one, J_B, g_eff, Ns, z)
        H_4 = build_hamiltonian_phi(mu, U, delta_S2_4, C_B_4, t_phi_4, beta_4, n_op, n_n_minus_one, J_B, g_eff, Ns, z)
        

        eigvals_1, eigvecs_1 = eigh(H_1)
        eigvals_2, eigvecs_2 = eigh(H_2)
        eigvals_3, eigvecs_3 = eigh(H_3)
        eigvals_4, eigvecs_4 = eigh(H_4)

        gs_1 = eigvecs_1[:, 0]
        gs_2 = eigvecs_2[:, 0]
        gs_3 = eigvecs_3[:, 0]
        gs_4 = eigvecs_4[:, 0]
        
        new_psi_1 = np.vdot(gs_1, np.dot(a_op, gs_1))
        new_psi_2 = np.vdot(gs_2, np.dot(a_op, gs_2))
        new_psi_3 = np.vdot(gs_3, np.dot(a_op, gs_3))
        new_psi_4 = np.vdot(gs_4, np.dot(a_op, gs_4))

        new_rho_1 = np.vdot(gs_1, np.dot(n_op, gs_1)).real
        new_rho_2 = np.vdot(gs_2, np.dot(n_op, gs_2)).real
        new_rho_3 = np.vdot(gs_3, np.dot(n_op, gs_3)).real
        new_rho_4 = np.vdot(gs_4, np.dot(n_op, gs_4)).real

        energy_1 = np.vdot(gs_1, np.dot(H_1, gs_1))
        energy_2 = np.vdot(gs_2, np.dot(H_2, gs_2))
        energy_3 = np.vdot(gs_3, np.dot(H_3, gs_3))
        energy_4 = np.vdot(gs_4, np.dot(H_4, gs_4))
        energy = (energy_1 + energy_2 + energy_3 + energy_4)/4
        
        if max(abs(new_psi_1 - psi_1), 
               abs(new_psi_2 - psi_2), 
               abs(new_psi_3 - psi_3), 
               abs(new_psi_4 - psi_4),
               abs(new_rho_1 - rho_1), 
               abs(new_rho_2 - rho_2), 
               abs(new_rho_3 - rho_3), 
               abs(new_rho_4 - rho_4)) < 1e-4:
            psi_1, psi_2, psi_3, psi_4, rho_1, rho_2, rho_3, rho_4 = new_psi_1, new_psi_2, new_psi_3, new_psi_4, new_rho_1, new_rho_2, new_rho_3, new_rho_4
            break

        psi_1, psi_2, psi_3, psi_4 = new_psi_1, new_psi_2, new_psi_3, new_psi_4
        rho_1, rho_2, rho_3, rho_4 = new_rho_1, new_rho_2, new_rho_3, new_rho_4

    density = (rho_1 + rho_2 + rho_3 + rho_4) / 4.0
    return psi_1, psi_2, psi_3, psi_4, rho_1, rho_2, rho_3, rho_4, density, energy, eigvecs_1, eigvecs_2, eigvecs_3, eigvecs_4


def plot_phase_diagram(t0_range, mu_range, resolution, N_cond=15):

    t0_values = np.linspace(*t0_range, resolution)
    mu_values = np.linspace(*mu_range, resolution)

    total_psi_matrix = np.zeros((resolution, resolution))
    density_matrix = np.zeros((resolution, resolution)) 
    dimer_imbalance_matrix = np.zeros((resolution, resolution)) 

    phase1_matrix = np.zeros((resolution, resolution))
    phase2_matrix = np.zeros((resolution, resolution))
    phase3_matrix = np.zeros((resolution, resolution))
    phase4_matrix = np.zeros((resolution, resolution))

    for i, t0 in enumerate(t0_values):
        for j, mu in enumerate(mu_values):
            min_energy = np.inf
            best_total_psi = 0
            best_density = 0
            best_imbalance = 0

            # Se prueban varias condiciones iniciales y configuración de fase para buscar el mínimo de energía
            candidate_phase_configs = [
                [0, 0, 0, 0],
                [np.pi, 0, np.pi, 0],
                [0, np.pi, 0, np.pi]
            ]
            
            for phase_config in candidate_phase_configs:
                for _ in range(N_cond):
                    psi_1_0 = np.random.uniform(0, 0.5) * np.exp(1j * phase_config[0])
                    psi_2_0 = np.random.uniform(0, 0.5) * np.exp(1j * phase_config[1])
                    psi_3_0 = np.random.uniform(0, 0.5) * np.exp(1j * phase_config[2])
                    psi_4_0 = np.random.uniform(0, 0.5) * np.exp(1j * phase_config[3])

                    rho_1_0 = np.random.uniform(0, 0.9)
                    rho_2_0 = np.random.uniform(0, 0.9)
                    rho_3_0 = np.random.uniform(0, 0.9)
                    rho_4_0 = np.random.uniform(0, 0.9)
                    
                    psi_1, psi_2, psi_3, psi_4, rho_1, rho_2, rho_3, rho_4, density, energy, eigvecs_1, eigvecs_2, eigvecs_3, eigvecs_4 = fixed_point_iteration(mu, t0, psi_1_0, psi_2_0, psi_3_0, psi_4_0, rho_1_0, rho_2_0, rho_3_0, rho_4_0, max_iters=300)
                    
                    if energy < min_energy:
                        min_energy = energy
                        total_psi =  (abs(psi_1) + abs(psi_2) + abs(psi_3 + abs(psi_4))) / 4 
                        imbalance = (rho_1 + rho_2 - rho_3 - rho_4) / 4
                        best_total_psi = total_psi
                        best_imbalance = abs(imbalance)
                        best_psi_1, best_psi_2, best_psi_3, best_psi_4 = psi_1, psi_2, psi_3, psi_4
                        best_rho_1, best_rho_2, best_rho_3, best_rho_4 = rho_1, rho_2, rho_3, rho_4

            print(f"({z*t0:.2f},{mu:.2f}): psi = ({np.abs(best_psi_1):.1f}ei {int(np.angle(best_psi_1))}, {np.abs(best_psi_2):.1f}ei {int(np.angle(best_psi_2))}, {np.abs(best_psi_3):.1f}ei {int(np.angle(best_psi_3))}, {np.abs(best_psi_4):.1f}ei {int(np.angle(best_psi_4))}), rho = ({best_rho_1:.2f}, {best_rho_2:.2f}, {best_rho_3:.2f}, {best_rho_4:.2f}) => im = {best_imbalance:.2f}, sf = {best_total_psi:.2f}")

            total_psi_matrix[j, i] = best_total_psi
            dimer_imbalance_matrix[j, i] = best_imbalance
            phase1_matrix[j, i] = np.angle(best_psi_1)
            phase2_matrix[j, i] = np.angle(best_psi_2)
            phase3_matrix[j, i] = np.angle(best_psi_3)
            phase4_matrix[j, i] = np.angle(best_psi_4)

    # Gráfica de total_psi
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = Normalize(vmin=np.min(total_psi_matrix), vmax=np.max(total_psi_matrix))
    cbar = ScalarMappable(norm=norm, cmap='viridis')
    c = ax.pcolormesh(z*t0_values, mu_values, total_psi_matrix, shading='auto', norm=norm)
    ax.set_xlabel('zt0', fontsize=14)
    ax.set_ylabel('mu', fontsize=14)
    ax.set_title('total_psi', fontsize=16)
    fig.colorbar(c, ax=ax, label='total_psi')
    plt.show()

    # Gráfica de imbalance
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = Normalize(vmin=np.min(dimer_imbalance_matrix), vmax=np.max(dimer_imbalance_matrix))
    cbar = ScalarMappable(norm=norm, cmap='viridis')
    c = ax.pcolormesh(z*t0_values, mu_values, dimer_imbalance_matrix, shading='auto', norm=norm)
    ax.set_xlabel('zt0', fontsize=14)
    ax.set_ylabel('mu', fontsize=14)
    ax.set_title('Dimer Imbalance', fontsize=16)
    fig.colorbar(c, ax=ax, label='Dimer Imbalance')
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    matrices = [phase1_matrix, phase2_matrix, phase3_matrix, phase4_matrix]
    títulos = [
        r'Fase de $\psi_1$',
        r'Fase de $\psi_2$',
        r'Fase de $\psi_3$',
        r'Fase de $\psi_4$'
    ]

    for ax, mat, title in zip(axes.ravel(), matrices, títulos):
        norm = Normalize(vmin=-np.pi, vmax=np.pi)
        pcm = ax.pcolormesh(z*t0_values, mu_values, mat, shading='auto', norm=norm, cmap='twilight')
        ax.set_xlabel('zt0', fontsize=12)
        ax.set_ylabel('μ', fontsize=12)
        ax.set_title(title, fontsize=14)
        cbar = fig.colorbar(pcm, ax=ax, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    plt.show()
    

def multipoints():
    zt_range = (0, .25/z)
    mu_range = (0, 2.5)
    resolution = 30
    plot_phase_diagram(zt_range, mu_range, resolution)


multipoints()