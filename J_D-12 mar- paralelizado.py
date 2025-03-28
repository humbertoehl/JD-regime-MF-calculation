import numpy as np
from scipy.linalg import eigh
import time
from multiprocessing import Pool, cpu_count


# --------------------------------------------------------------------
def create_fock_operators(dim):
    b = np.zeros((dim, dim))
    for n in range(1, dim):
        b[n-1, n] = np.sqrt(n)
    b_dag = b.T
    n_op = np.dot(b_dag, b)
    I = np.eye(dim)
    return b, b_dag, n_op, I

# --------------------------------------------------------------------
def mean_field_iteration(initial_params, params, b, b_dag, n_op, I, tol=1e-5, max_iter=150):
    psi_o = initial_params['psi_o']
    psi_e = initial_params['psi_e']
    rho_o = initial_params['rho_o']
    rho_e = initial_params['rho_e']
    
    E_odd = None
    E_even = None

    for it in range(max_iter):
        Delta_rho = (rho_o - rho_e) / 2.0
        
        mu_o = params['mu'] + 2 * params['g_eff'] * params['N_s'] * (params['J_D']**2) * Delta_rho
        mu_e = params['mu'] - 2 * params['g_eff'] * params['N_s'] * (params['J_D']**2) * Delta_rho
        
        U_eff = params['U'] + 2 * params['g_eff'] * (params['J_D']**2)
        
        C_D_o = (params['N_s'] * (params['J_D']**2) * Delta_rho * rho_o) / 2.0 - ((params['J_D']**2) * rho_o**2) / 2.0
        C_D_e = (-params['N_s'] * (params['J_D']**2) * Delta_rho * rho_e) / 2.0 - ((params['J_D']**2) * rho_e**2) / 2.0
        
        # Desacoplar beta según la formulación: 
        # Para la sublattice odd:
        beta_odd = psi_e * (b + b_dag) - 2 * psi_e * psi_o
        # Para la sublattice even:
        beta_even = psi_o * (b + b_dag) - 2 * psi_e * psi_o
        
        fac = params['N_s'] / 2.0
        
        H_odd = fac * ( - params['z'] * params['t0'] * beta_odd
                        - mu_o * n_op 
                        + (U_eff / 2.0) * (np.dot(n_op, n_op) - n_op) 
                        - params['g_eff'] * (params['J_D']**2) * rho_o * n_op 
                        - params['g_eff'] * C_D_o * np.eye(n_op.shape[0]) )
        
        H_even = fac * ( - params['z'] * params['t0'] * beta_even
                         - mu_e * n_op 
                         + (U_eff / 2.0) * (np.dot(n_op, n_op) - n_op) 
                         - params['g_eff'] * (params['J_D']**2) * rho_e * n_op 
                         - params['g_eff'] * C_D_e * np.eye(n_op.shape[0]) )
        
        # Diagonalización de los Hamiltonianos
        E_vals_odd, E_vecs_odd = eigh(H_odd)
        E_vals_even, E_vecs_even = eigh(H_even)
        
        idx_odd = np.argmin(E_vals_odd)
        idx_even = np.argmin(E_vals_even)
        psi_state_odd = E_vecs_odd[:, idx_odd]
        psi_state_even = E_vecs_even[:, idx_even]
        
        E_odd = E_vals_odd[idx_odd]
        E_even = E_vals_even[idx_even]
        
        # Recalcular parámetros de campo medio
        psi_o_new = np.vdot(psi_state_odd, np.dot(b, psi_state_odd))
        rho_o_new = np.vdot(psi_state_odd, np.dot(n_op, psi_state_odd))
        psi_e_new = np.vdot(psi_state_even, np.dot(b, psi_state_even))
        rho_e_new = np.vdot(psi_state_even, np.dot(n_op, psi_state_even))
        
        diff = np.abs(np.array([psi_o_new - psi_o, psi_e_new - psi_e, rho_o_new - rho_o, rho_e_new - rho_e]))
        if np.all(diff < tol):
            psi_o, psi_e, rho_o, rho_e = psi_o_new, psi_e_new, rho_o_new, rho_e_new
            break
        
        psi_o, psi_e, rho_o, rho_e = psi_o_new, psi_e_new, rho_o_new, rho_e_new
        
    final_params = {'psi_o': psi_o, 'psi_e': psi_e, 'rho_o': rho_o, 'rho_e': rho_e}
    energies = {'E_odd': E_odd, 'E_even': E_even}
    return final_params, energies

# --------------------------------------------------------------------
def run_mean_field(initial_conditions, params, b, b_dag, n_op, I, tol=1e-5, max_iter=150):
    final_params, energies = mean_field_iteration(initial_conditions, params, b, b_dag, n_op, I, tol, max_iter)
    total_energy = energies['E_odd'] + energies['E_even']
    return final_params, total_energy

# --------------------------------------------------------------------
# Función que computa un único punto de la grilla (mu, zt)
def compute_grid_point(args):
    mu_val, zt_val, base_params, psi_vals_e, psi_vals_o, rho_vals, b, b_dag, n_op, I, tol, max_iter = args
    # Copia local de parámetros y actualización de mu y t0
    params_local = base_params.copy()
    params_local['mu'] = mu_val
    params_local['t0'] = zt_val / params_local['z']
    
    best_energy = np.inf
    best_solution = None
    best_cond = None
    for psi_o_init in psi_vals_o:
        for psi_e_init in psi_vals_e:
            for rho_o_init in rho_vals:
                for rho_e_init in rho_vals:
                    if rho_o_init==rho_e_init:
                        continue
                    init_cond = {
                        'psi_o': psi_o_init,
                        'psi_e': psi_e_init,
                        'rho_o': rho_o_init,
                        'rho_e': rho_e_init
                    }
                    sol, energy = run_mean_field(init_cond, params_local, b, b_dag, n_op, I, tol, max_iter)
                    if energy < best_energy:
                        best_energy = energy
                        best_solution = sol.copy()
                        best_cond = init_cond.copy()
    psi_total = (np.abs(best_solution['psi_o']) + np.abs(best_solution['psi_e'])) / 2.0
    delta_rho = abs(best_solution['rho_o'] - best_solution['rho_e']) / 2.0
    print(f'(mu,zt)=({mu_val:.4f},{zt_val:.4f}) -> SF={psi_total:.4f}, Imbalance={delta_rho:.4f}, init={(best_cond)}')
    return (mu_val, zt_val, best_solution['psi_o'], best_solution['psi_e'],
            best_solution['rho_o'], best_solution['rho_e'], psi_total, delta_rho, best_cond)

# --------------------------------------------------------------------
def sweep_phase_diagram_parallel(mu_vals, zt_vals, base_params, b, b_dag, n_op, I,
                                 psi_vals_e, psi_vals_o, rho_vals, tol=1e-5, max_iter=150):
    args_list = []
    for mu_val in mu_vals:
        for zt_val in zt_vals:
            args_list.append((mu_val, zt_val, base_params, psi_vals_e, psi_vals_o,
                              rho_vals, b, b_dag, n_op, I, tol, max_iter))
    
    pool = Pool(processes=cpu_count())
    results = pool.map(compute_grid_point, args_list)
    pool.close()
    pool.join()
    
    # Preparar lista de filas para guardar en el txt
    data_rows = []
    for res in results:
        mu_val, zt_val, psi_o, psi_e, rho_o, rho_e, psi_total, delta_rho, best_cond = res
        data_rows.append([mu_val, zt_val, psi_o, psi_e, rho_o, rho_e, psi_total, delta_rho])
    
    return data_rows

# --------------------------------------------------------------------
if __name__ == '__main__':
    # Parámetros base
    base_params = {
        'U': 1.0,
        'J_D': 1.0,
        'N_s': 100,
        'z': 6,
        'mu': 0.5,       # se actualizará
        't0': 0.05 / 6.0 # se actualizará
    }
    base_params['g_eff'] = -0.5 * base_params['U'] / base_params['N_s']  # -0.005
    
    # Operadores en el espacio de Fock
    dim = 31
    b, b_dag, n_op, I = create_fock_operators(dim)
    
    # Grilla de parámetros
    N_mu = 30
    N_zt = 30
    mu_vals = np.linspace(0.5, 1.4, N_mu)
    zt_vals = np.linspace(0.05, 0.17, N_zt)
    
    # Conjuntos de condiciones iniciales (se proponen rangos más amplios basados en intuición)
    psi_vals_e = [0.1,0.01,0.001,0.5]
    psi_vals_o = [0.2,0.02,0.002,0.6]
    rho_vals = [0, 0.2, 0.4, 0.6, 0.8]
    
    # Ejecutar en paralelo
    start = time.time()
    data_rows = sweep_phase_diagram_parallel(mu_vals, zt_vals, base_params, b, b_dag, n_op, I,
                                               psi_vals_e, psi_vals_o, rho_vals, tol=1e-5, max_iter=150)
    end = time.time()
    time_of_excecution = (end - start)/60
    print(f'{time_of_excecution:.2f} min')


    #Guardar Resultados
    output_filename = "JD-Results(12).txt"
    with open(output_filename, "w") as f:
        f.write("# Resultados del barrido de la grilla en (mu, zt/U)\n")
        f.write("# Columnas: mu, zt/U, psi_o, psi_e, rho_o, rho_e, psi_total, delta_rho\n")
        f.write("# Metadatos:\n")
        f.write(f"# base_params: {base_params}\n")
        f.write(f"# dim: {dim}\n")
        f.write(f"# psi_vals_o: {psi_vals_o}\n")
        f.write(f"# psi_vals_e: {psi_vals_e}\n")
        f.write(f"# rho_vals: {rho_vals}\n")
        f.write(f"# Tiempo de ejecución: {time_of_excecution} min\n")
        f.write("#\n")
        f.write("#mu\tzt/U\tpsi_o\tpsi_e\trho_o\trho_e\tpsi_total\tdelta_rho\n")

        for row in data_rows:
            f.write("\t".join([f"{x:.6f}" for x in row]) + "\n")
    
    print(f"Archivo de resultados guardado en: {output_filename}")
