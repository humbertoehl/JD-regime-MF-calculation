import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Parámetros globales
Ns    = 100
z     = 6
J_B   = 0.05
U     = 1.0
g_eff = -0.25
n_max = 5
resolution = 20
N_cond = 16

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

def cross(psi_i, psi_j, sqrd):
    if sqrd is False:
        return (psi_i.conjugate()*psi_j + psi_j.conjugate()*psi_i)
    if sqrd is True:
        return (psi_i.conjugate()*psi_j + psi_j.conjugate()*psi_i)**2 * Id

def create_beta_op(psi_local, psi_neighbor):
    return  psi_neighbor * (a_op + adag_op - psi_local * Id)

def beta_local(xi, psi):
    # psi: lista [psi1,psi2,psi3,psi4]
    ip1 = (xi % 4)
    psi_i, psi_ip1 = psi[xi], psi[ip1]
    return (psi_i.conjugate()*a_op + psi_ip1.conjugate()*a_op
            + psi_ip1*adag_op + psi_i*adag_op
            - (psi_i.conjugate()*psi_ip1 + (psi_i.conjugate()*psi_ip1).conjugate())*Id)

def build_hamiltonian(t_xi, mu, beta_xi, delta_s2_xi, tilde_c_xi):
    
    term_hopping  = -(z/2) * t_xi * beta_xi
    term_mu       = - mu * n_op
    term_U        = (U/2) * n_n_minus_one
    term_delta_s2 = g_eff *  J_B**2 * delta_s2_xi *0
    term_const    = - g_eff * Ns * J_B**2 * tilde_c_xi * Id
    H_local = term_hopping + term_mu + term_U + term_delta_s2 + term_const
    return H_local * Ns/4.0


def create_delsa_s2_op(beta_op, psi_i, psi_ip1, psi_im1, rho_i, rho_ip1, rho_im1):

    # combinación de fases vecinos (+1 y −1)
    lam_p = psi_i.conjugate()*psi_ip1 + psi_ip1.conjugate()*psi_i
    lam_m = psi_i.conjugate()*psi_im1 + psi_im1.conjugate()*psi_i

    # <b^2> y su conjugado para vecinos
    mean_b2_p = psi_ip1**2
    mean_b2_p_c = (psi_ip1.conjugate())**2
    mean_b2_m = psi_im1**2
    mean_b2_m_c = (psi_im1.conjugate())**2

    # — vecino i+1 —
    # 1) b_i^†2 b_{i+1}^2 + b_{i+1}^†2 b_i^2
    term_b2_p = (adag_op @ adag_op) * mean_b2_p \
              + (a_op @ a_op)   * mean_b2_p_c \
              - (mean_b2_p_c * mean_b2_p) * Id

    # 2) 2 n_i n_{i+1}
    term_nn_p = 2 * (n_op * rho_ip1) \
              - 2 * (rho_i * rho_ip1) * Id

    # 3) n_i + n_{i+1}
    term_lin_p = n_op + rho_ip1 * Id

    # 4) −2 (ψ_i*ψ_{i+1}+c.c.) β_i
    term_beta_p = -2 * lam_p * beta_op

    # 5) (ψ_i*ψ_{i+1}+c.c.)^2  (constante)
    term_const_p = (lam_p**2) * Id

    # — vecino i−1 (análogos) —
    term_b2_m = (adag_op @ adag_op) * mean_b2_m \
              + (a_op @ a_op)   * mean_b2_m_c \
              - (mean_b2_m_c * mean_b2_m) * Id

    term_nn_m = 2 * (n_op * rho_im1) \
              - 2 * (rho_i * rho_im1) * Id

    term_lin_m = n_op + rho_im1 * Id

    term_beta_m = -2 * lam_m * beta_op

    term_const_m = (lam_m**2) * Id

    # sumatorio de todos los trozos
    delta_s2 = (
        term_b2_p + term_nn_p + term_lin_p + term_beta_p + term_const_p
      + term_b2_m + term_nn_m + term_lin_m + term_beta_m + term_const_m
    )

    # factor global z/4
    return (z/4) * delta_s2

    

def fixed_point_iteration(t0, mu, psi_1, psi_2, psi_3, psi_4, rho_1, rho_2, rho_3, rho_4, max_iters=200):
    for i in range(max_iters):

        sum_eta = (psi_1.conjugate()*psi_2 + psi_2.conjugate()*psi_1) - (psi_2.conjugate()*psi_3 + psi_3.conjugate()*psi_2) + (psi_3.conjugate()*psi_4 + psi_4.conjugate()*psi_3) - (psi_4.conjugate()*psi_1 + psi_1.conjugate()*psi_4)

        tilde_eta_1 = (z/8) *sum_eta
        tilde_eta_2 = -(z/8) *sum_eta
        tilde_eta_3 = (z/8) *sum_eta
        tilde_eta_4 = -(z/8) *sum_eta

        t_1 = t0 - g_eff * (Ns) * J_B**2 * tilde_eta_1
        t_2 = t0 - g_eff * (Ns) * J_B**2 * tilde_eta_2
        t_3 = t0 - g_eff * (Ns) * J_B**2 * tilde_eta_3
        t_4 = t0 - g_eff * (Ns) * J_B**2 * tilde_eta_4

        tilde_c_1 = (z/4) * (psi_1.conjugate()*psi_2 +psi_2.conjugate()*psi_1) * tilde_eta_1
        tilde_c_2 = (z/4) * (psi_2.conjugate()*psi_3 +psi_3.conjugate()*psi_2) * tilde_eta_2
        tilde_c_3 = (z/4) * (psi_3.conjugate()*psi_4 +psi_4.conjugate()*psi_3) * tilde_eta_3
        tilde_c_4 = (z/4) * (psi_4.conjugate()*psi_1 +psi_1.conjugate()*psi_4) * tilde_eta_4
        

        psi = [psi_1,psi_2,psi_3,psi_4]
        beta_1 = beta_local(0, psi)
        beta_2 = beta_local(1, psi)
        beta_3 = beta_local(2, psi)
        beta_4 = beta_local(3, psi)




        delta_s2_1 = create_delsa_s2_op(beta_1, psi_1, psi_2, psi_4, rho_1, rho_2, rho_4)
        delta_s2_2 = create_delsa_s2_op(beta_2, psi_2, psi_3, psi_1, rho_2, rho_3, rho_1)
        delta_s2_3 = create_delsa_s2_op(beta_3, psi_3, psi_4, psi_2, rho_3, rho_4, rho_2)
        delta_s2_4 = create_delsa_s2_op(beta_4, psi_4, psi_1, psi_3, rho_4, rho_1, rho_3)

        H_1 = build_hamiltonian(t_1, mu, beta_1, delta_s2_1, tilde_c_1)
        H_2 = build_hamiltonian(t_2, mu, beta_2, delta_s2_2, tilde_c_2)
        H_3 = build_hamiltonian(t_3, mu, beta_3, delta_s2_3, tilde_c_3)
        H_4 = build_hamiltonian(t_4, mu, beta_4, delta_s2_4, tilde_c_4)
        
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

        energy_1    = np.vdot(gs_1, np.dot(H_1, gs_1))
        energy_2    = np.vdot(gs_2, np.dot(H_2, gs_2))
        energy_3    = np.vdot(gs_3, np.dot(H_3, gs_3))
        energy_4    = np.vdot(gs_4, np.dot(H_4, gs_4))

        energy = energy_1 + energy_2 + energy_3 + energy_4
        
        if max(abs(new_psi_1 - psi_1), abs(new_psi_2 - psi_2),
               abs(new_psi_3 - psi_3), abs(new_psi_4 - psi_4),
               abs(new_rho_1 - rho_1), abs(new_rho_2 - rho_2),
               abs(new_rho_3 - rho_3), abs(new_rho_4 - rho_4)) < 1e-3:
            psi_1, psi_2, psi_3, psi_4 = new_psi_1, new_psi_2, new_psi_3, new_psi_4
            rho_1, rho_2, rho_3, rho_4 = new_rho_1, new_rho_2, new_rho_3, new_rho_4
            break

        psi_1, psi_2, psi_3, psi_4 = new_psi_1, new_psi_2, new_psi_3, new_psi_4
        rho_1, rho_2, rho_3, rho_4 = new_rho_1, new_rho_2, new_rho_3, new_rho_4

    return psi_1, psi_2, psi_3, psi_4,rho_1, rho_2, rho_3, rho_4, energy

def plot_phase_diagram(t0_range, mu_range, resolution, N_cond=N_cond):
    t0_values = np.linspace(*t0_range, resolution)
    mu_values = np.linspace(*mu_range, resolution)

    total_psi_matrix = np.zeros((resolution, resolution))
    dimer_imbalance_matrix = np.zeros((resolution, resolution))
    dimer_imbalance_matrix2 = np.zeros((resolution, resolution))
    avg_density_matrix = np.zeros((resolution, resolution))
    phase_diff_matrix = np.zeros((resolution, resolution))
    phase_diff_matrix2 = np.zeros((resolution, resolution))

    for i, t0 in enumerate(t0_values):
        for j, mu in enumerate(mu_values):

            min_energy = np.inf
            best_total_psi = None
            best_dimer_imbalance = None
            best_dimer_imbalance2 = None
            best_avg_density = None
            best_phase_diff = None
            best_phase_diff2 = None

            # Se prueban varias condiciones iniciales para buscar el mínimo de energía. Paso esencial, pues el sistema tiene muchos mínimos locales
            for _ in range(N_cond):
                sign = np.random.choice([1,-1])
                psi_1_0 = (np.random.uniform(0, 0.4) + 0.0001j)* sign
                psi_2_0 = (np.random.uniform(0, 0.4) + 0.0001j)* sign
                psi_3_0 = (np.random.uniform(0, 0.4) + 0.0001j)* sign * (-1)
                psi_4_0 = (np.random.uniform(0, 0.4) + 0.0001j)* sign * (-1)

                rho_1_0 = np.random.uniform(0, 1.3)
                rho_2_0 = np.random.uniform(0, 1.3)
                rho_3_0 = np.random.uniform(0, 1.3)
                rho_4_0 = np.random.uniform(0, 1.3)
                
                psi_1, psi_2, psi_3, psi_4,rho_1, rho_2, rho_3, rho_4, energy = fixed_point_iteration(t0, mu, psi_1_0, psi_2_0, psi_3_0, psi_4_0, rho_1_0, rho_2_0, rho_3_0, rho_4_0)
                
                if energy < min_energy:
                    min_energy = energy

                    best_total_psi = (np.abs(psi_1)+np.abs(psi_2)+np.abs(psi_3)+np.abs(psi_4))/4
                    best_dimer_imbalance = abs(rho_1 + rho_2 - rho_3 - rho_4)/4
                    best_dimer_imbalance2 = abs(rho_1 + rho_3 - rho_1 - rho_4)/4
                    best_avg_density = (rho_1 + rho_2 + rho_3 + rho_4)/4
                    best_phase_diff = abs(np.angle(psi_3)+np.angle(psi_4)-np.angle(psi_2)-np.angle(psi_1))/2
                    best_phase_diff2 = abs(np.angle(psi_4)+np.angle(psi_2)-np.angle(psi_3)-np.angle(psi_1))/2
                    if abs(best_phase_diff-2*np.pi)<.1:
                        best_phase_diff=0

            print(f"({z*t0:.2f},{mu:.2f}) => psi_t = {best_total_psi:.2f}, Drho = {best_dimer_imbalance:.2f}, rho_avg = {best_avg_density:.2f}, DPhi = {best_phase_diff:.2f}")
            #print(f"{np.real(psi_1):.2f} {np.angle(psi_1):.1f},{np.real(psi_2):.2f} {np.angle(psi_2):.1f},{np.real(psi_3):.2f} {np.angle(psi_3):.1f},{np.real(psi_4):.2f} {np.angle(psi_4):.1f} | {rho_1:.2f},{rho_2:.2f},{rho_3:.2f},{rho_4:.2f}")
            print(f"densidades: ({rho_1:.3f}) ({rho_2:.3f}) ({rho_3:.3f}) ({rho_4:.3f})")
            #print(f"Fases: ({np.angle(psi_1):.1f}) ({np.angle(psi_2):.1f}) ({np.angle(psi_3):.1f}) ({np.angle(psi_4):.1f})")
            total_psi_matrix[j, i] = best_total_psi
            dimer_imbalance_matrix[j, i] = best_dimer_imbalance
            dimer_imbalance_matrix2[j, i] = best_dimer_imbalance2
            avg_density_matrix[j, i] = best_avg_density
            phase_diff_matrix[j, i] = best_phase_diff
            phase_diff_matrix2[j, i] = best_phase_diff2

    # Gráfica de total_psi
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = Normalize(vmin=np.min(total_psi_matrix), vmax=np.max(total_psi_matrix))
    cbar = ScalarMappable(norm=norm, cmap='viridis')
    c = ax.pcolormesh(z*t0_values, mu_values, total_psi_matrix, shading='auto', norm=norm)
    ax.set_xlabel('zt0', fontsize=14)
    ax.set_ylabel('mu', fontsize=14)
    ax.set_title('total_psi', fontsize=16)
    fig.colorbar(c, ax=ax, label='total_psi')

    # Gráfica de dimer_imbalance
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = Normalize(vmin=np.min(dimer_imbalance_matrix), vmax=np.max(dimer_imbalance_matrix))
    cbar = ScalarMappable(norm=norm, cmap='viridis')
    c = ax.pcolormesh(z*t0_values, mu_values, dimer_imbalance_matrix, shading='auto', norm=norm)
    ax.set_xlabel('zt0', fontsize=14)
    ax.set_ylabel('mu', fontsize=14)
    ax.set_title('dimer_imbalance', fontsize=16)
    fig.colorbar(c, ax=ax, label='dimer_imbalance')

        # Gráfica de dimer_imbalance2
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = Normalize(vmin=np.min(dimer_imbalance_matrix2), vmax=np.max(dimer_imbalance_matrix2))
    cbar = ScalarMappable(norm=norm, cmap='viridis')
    c = ax.pcolormesh(z*t0_values, mu_values, dimer_imbalance_matrix2, shading='auto', norm=norm)
    ax.set_xlabel('zt0', fontsize=14)
    ax.set_ylabel('mu', fontsize=14)
    ax.set_title('dimer_imbalance2', fontsize=16)
    fig.colorbar(c, ax=ax, label='dimer_imbalance2')

    # Gráfica de  avg_density
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = Normalize(vmin=np.min( avg_density_matrix), vmax=np.max( avg_density_matrix))
    cbar = ScalarMappable(norm=norm, cmap='viridis')
    c = ax.pcolormesh(z*t0_values, mu_values,  avg_density_matrix, shading='auto', norm=norm)
    ax.set_xlabel('zt0', fontsize=14)
    ax.set_ylabel('mu', fontsize=14)
    ax.set_title(' avg_density', fontsize=16)
    fig.colorbar(c, ax=ax, label=' avg_density')

    # Gráfica de phase_diff
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = Normalize(vmin=np.min(phase_diff_matrix), vmax=np.max(phase_diff_matrix))
    cbar = ScalarMappable(norm=norm, cmap='viridis')
    c = ax.pcolormesh(z*t0_values, mu_values, phase_diff_matrix, shading='auto', norm=norm)
    ax.set_xlabel('zt0', fontsize=14)
    ax.set_ylabel('mu', fontsize=14)
    ax.set_title('phase_diff', fontsize=16)
    fig.colorbar(c, ax=ax, label='phase_diff')

    # Gráfica de phase_diff
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = Normalize(vmin=np.min(phase_diff_matrix2), vmax=np.max(phase_diff_matrix2))
    cbar = ScalarMappable(norm=norm, cmap='viridis')
    c = ax.pcolormesh(z*t0_values, mu_values, phase_diff_matrix2, shading='auto', norm=norm)
    ax.set_xlabel('zt0', fontsize=14)
    ax.set_ylabel('mu', fontsize=14)
    ax.set_title('phase_diff2', fontsize=16)
    fig.colorbar(c, ax=ax, label='phase_diff2')
    plt.show()












def _density_for_mu(t0, mu, N_cond=5):
    min_energy = np.inf
    best_total_psi = None
    best_dimer_imbalance = None
    best_avg_density = None
    best_phase_diff = None

    for _ in range(N_cond):
        signo_de_fase = np.random.choice([0,1])
        psi_1_0 = np.random.uniform(0, 0.3) +1e-5j
        psi_2_0 = np.random.uniform(0, 0.3) +1e-5j
        psi_3_0 = np.random.uniform(0, 0.3)*signo_de_fase +1e-5j
        psi_4_0 = np.random.uniform(0, 0.3)*signo_de_fase +1e-5j

        rho_1_0 = np.random.uniform(0, 1.4)
        rho_2_0 = np.random.uniform(0, 1.4)
        rho_3_0 = np.random.uniform(0, 1.4)
        rho_4_0 = np.random.uniform(0, 1.4)

        psi_1, psi_2, psi_3, psi_4,rho_1, rho_2, rho_3, rho_4, energy = fixed_point_iteration(t0, mu, psi_1_0, psi_2_0, psi_3_0, psi_4_0, rho_1_0, rho_2_0, rho_3_0, rho_4_0)

        if energy < min_energy:
            min_energy = energy

            best_total_psi = (np.abs(psi_1)+np.abs(psi_2)+np.abs(psi_3)+np.abs(psi_4))/4
            best_dimer_imbalance = (rho_1 + rho_2 - rho_3 - rho_4)/4
            best_avg_density = (rho_1 + rho_2 + rho_3 + rho_4)/4
            best_phase_diff = abs(np.angle(psi_3)-np.angle(psi_1))
            if abs(best_phase_diff-2*np.pi)<.1:
                best_phase_diff=0

    return best_total_psi, best_dimer_imbalance, best_avg_density, best_phase_diff


def _find_mu_for_density(t0, rho_target, mu_min=-1.0, mu_max=3.0, tol=1e-1, max_iter=8, N_cond=6):

    # Valor de densidad en los extremos
    _, _, rho_lo, _ = _density_for_mu(t0, mu_min, N_cond)
    _, _, rho_hi, _ = _density_for_mu(t0, mu_max, N_cond)

    # Comprobamos que haya encierro
    if (rho_lo - rho_target) * (rho_hi - rho_target) > 0:
        return None, None, None, None, None   # No hay raíz en el intervalo

    for _ in range(max_iter):
        mu_mid = 0.5 * (mu_min + mu_max)
        total_psi_mid, imbalance_mid, rho_mid, phase_diff_mid = _density_for_mu(t0, mu_mid, N_cond)

        if rho_mid is None:
            return None, None, None, None, None  # falló cálculo intermedio

        if abs(rho_mid - rho_target) < tol:
            return mu_mid, total_psi_mid, imbalance_mid, rho_mid, phase_diff_mid

        # Bisección
        if (rho_lo - rho_target) * (rho_mid - rho_target) < 0:
            mu_max, rho_hi = mu_mid, rho_mid
        else:
            mu_min, rho_lo = mu_mid, rho_mid

    # Si llegó aquí, no convergió dentro de max_iter
    return None, None, None, None, None


def plot_phase_diagram_rho(t0_range, rho_range, resolution, mu_bounds=(-1.0, 3.0), N_cond=6):
    t0_values  = np.linspace(*t0_range,  resolution)
    rho_values = np.linspace(*rho_range, resolution)

    total_psi_matrix = np.zeros((resolution, resolution))
    dimer_imbalance_matrix = np.zeros((resolution, resolution))
    avg_density_matrix = np.zeros((resolution, resolution))
    phase_diff_matrix = np.zeros((resolution, resolution))

    for i, t0 in enumerate(t0_values):
        for j, rho_target in enumerate(rho_values):
            mu_opt, total_psi_opt, imbalance_opt, rho_opt, phase_diff_opt = _find_mu_for_density(t0, rho_target, mu_min=mu_bounds[0], mu_max=mu_bounds[1], tol=1e-3, max_iter=30, N_cond=N_cond)

            if mu_opt is None:
                # No se encontró solución; se deja cero por defecto
                print(f"[WARN] ρ={rho_target:.3f} no accesible para zt0={t0:.3f}")
                continue

            total_psi_matrix[j, i] = total_psi_opt
            dimer_imbalance_matrix[j, i] = imbalance_opt
            avg_density_matrix[j, i] = rho_opt
            phase_diff_matrix[j, i] = phase_diff_opt

            print(f"(zt0={t0:.2f}, ρ={rho_target:.2f}) -> μ={mu_opt:.4f}, ψ={total_psi_opt:.2f}, im={imbalance_opt:.2f}, rho={rho_opt:.2f}, phi={phase_diff_opt:.2f}")

    # ---------- Gráfica total_psi ----------
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    norm1 = Normalize(vmin=np.min(total_psi_matrix), vmax=np.max(total_psi_matrix))
    c1 = ax1.pcolormesh(z*t0_values, rho_values, total_psi_matrix, shading='auto', norm=norm1, cmap='viridis')
    ax1.set_xlabel('zt0', fontsize=14)
    ax1.set_ylabel('ρ (densidad promedio)', fontsize=14)
    ax1.set_title('total_ψ', fontsize=16)
    fig1.colorbar(c1, ax=ax1, label='total_ψ')

    # ---------- Gráfica dimer_imbalance ----------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    norm1 = Normalize(vmin=np.min(dimer_imbalance_matrix), vmax=np.max(dimer_imbalance_matrix))
    c1 = ax2.pcolormesh(z*t0_values, rho_values, dimer_imbalance_matrix, shading='auto', norm=norm1, cmap='viridis')
    ax2.set_xlabel('zt0', fontsize=14)
    ax2.set_ylabel('ρ (densidad promedio)', fontsize=14)
    ax2.set_title('dimer rimbalance', fontsize=16)
    fig2.colorbar(c1, ax=ax2, label='dimer rimbalance')

    # ---------- Gráfica avg_density ----------
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    norm1 = Normalize(vmin=np.min(avg_density_matrix), vmax=np.max(avg_density_matrix))
    c1 = ax3.pcolormesh(z*t0_values, rho_values, avg_density_matrix, shading='auto', norm=norm1, cmap='viridis')
    ax3.set_xlabel('zt0', fontsize=14)
    ax3.set_ylabel('ρ (densidad promedio)', fontsize=14)
    ax3.set_title('avg_density', fontsize=16)
    fig3.colorbar(c1, ax=ax3, label='avg_density')

    # ---------- Gráfica phase_diff ----------
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    norm1 = Normalize(vmin=np.min(phase_diff_matrix), vmax=np.max(phase_diff_matrix))
    c1 = ax4.pcolormesh(z*t0_values, rho_values, phase_diff_matrix, shading='auto', norm=norm1, cmap='viridis')
    ax4.set_xlabel('zt0', fontsize=14)
    ax4.set_ylabel('ρ (densidad promedio)', fontsize=14)
    ax4.set_title('phase_diff', fontsize=16)
    fig4.colorbar(c1, ax=ax4, label='phase_diff')


    plt.show()
    return fig1, fig2, fig3, fig4







def multipoints():
    t0_range = (0, .3/z)
    mu_range = (0, 3)
    rho_range = (0.1, 3)

    plot_phase_diagram(t0_range, mu_range, resolution)
    #plot_phase_diagram_rho(t0_range, rho_range, resolution, mu_bounds=(-2.0, 5.0), N_cond=8)


multipoints()
    