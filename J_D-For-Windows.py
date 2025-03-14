import os
import numpy as np
from scipy.linalg import eigh
import time
from multiprocessing import Pool, cpu_count

# Limitar BLAS a 1 hilo por proceso para evitar oversubscription
os.environ["OMP_NUM_THREADS"] = "1"

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
        beta_odd = psi_e * (b + b_dag) - 2 * psi_e * psi_o
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
# Función para calcular un único punto de la grilla.
# Si se proporciona 'warm_start', se usa esa condición inicial; de lo contrario se explora el espacio de condiciones.
def compute_single_point(mu_val, zt_val, base_params, b, b_dag, n_op, I, tol, max_iter,
                         psi_vals_e=None, psi_vals_o=None, rho_vals=None, warm_start=None):
    params_local = base_params.copy()
    params_local['mu'] = mu_val
    params_local['t0'] = zt_val / params_local['z']
    
    if warm_start is not None:
        init_cond = warm_start
        sol, energy = run_mean_field(init_cond, params_local, b, b_dag, n_op, I, tol, max_iter)
        best_solution = sol
        best_energy = energy
        best_cond = init_cond
    else:
        best_energy = np.inf
        best_solution = None
        best_cond = None
        for psi_o_init in psi_vals_o:
            for psi_e_init in psi_vals_e:
                for rho_o_init in rho_vals:
                    for rho_e_init in rho_vals:
                        if rho_o_init == rho_e_init:
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
    delta_rho = np.abs(best_solution['rho_o'] - best_solution['rho_e']) / 2.0
    print(f'done for {zt_val}-{mu_val}')
    return (mu_val, zt_val, best_solution['psi_o'], best_solution['psi_e'],
            best_solution['rho_o'], best_solution['rho_e'], psi_total, delta_rho, best_cond)

# --------------------------------------------------------------------
# Para cada fila (valor de μ), se recorre la lista de zt de forma secuencial, usando warm start.
def compute_mu_row(mu_val, zt_vals, base_params, psi_vals_e, psi_vals_o, rho_vals, b, b_dag, n_op, I, tol, max_iter):
    row_results = []
    warm_start = None  # Condición inicial para el primer punto
    for idx, zt_val in enumerate(zt_vals):
        if idx == 0:
            result = compute_single_point(mu_val, zt_val, base_params, b, b_dag, n_op, I, tol, max_iter,
                                          psi_vals_e=psi_vals_e, psi_vals_o=psi_vals_o, rho_vals=rho_vals,
                                          warm_start=None)
        else:
            # Uso de la solución anterior como condición inicial
            result = compute_single_point(mu_val, zt_val, base_params, b, b_dag, n_op, I, tol, max_iter,
                                          warm_start=warm_start)
        row_results.append(result)
        # Actualizar el warm start con la mejor condición encontrada en este punto
        _, _, _, _, _, _, _, _, best_cond = result
        warm_start = best_cond
    return row_results

# --------------------------------------------------------------------
# Barrido de la grilla en paralelo, paralelizando por cada fila (valor de μ).
def sweep_phase_diagram_parallel(mu_vals, zt_vals, base_params, b, b_dag, n_op, I,
                                 psi_vals_e, psi_vals_o, rho_vals, tol=1e-5, max_iter=150):
    args_list = []
    for mu_val in mu_vals:
        args_list.append((mu_val, zt_vals, base_params, psi_vals_e, psi_vals_o, rho_vals, b, b_dag, n_op, I, tol, max_iter))
    
    pool = Pool(processes=min(cpu_count(), len(mu_vals)))
    # Usamos starmap para pasar argumentos a compute_mu_row
    results = pool.starmap(compute_mu_row, args_list)
    pool.close()
    pool.join()
    
    # Aplanar los resultados (cada fila es una lista de puntos)
    data_rows = []
    for row in results:
        for res in row:
            mu_val, zt_val, psi_o, psi_e, rho_o, rho_e, psi_total, delta_rho, _ = res
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
    N_mu = 25
    N_zt = 25
    mu_vals = np.linspace(0.5, 1.4, N_mu)
    zt_vals = np.linspace(0.05, 0.17, N_zt)
    
    # Conjuntos de condiciones iniciales (usados solo para el primer punto de cada fila)
    psi_vals_e = [0.1, 0.01, 0.001, 0.5]
    psi_vals_o = [0.2, 0.02, 0.002, 0.6]
    rho_vals = [0, 0.2, 0.4, 0.6, 0.8]
    
    # Ejecutar el barrido en paralelo (por filas)
    start = time.time()
    data_rows = sweep_phase_diagram_parallel(mu_vals, zt_vals, base_params, b, b_dag, n_op, I,
                                               psi_vals_e, psi_vals_o, rho_vals, tol=1e-5, max_iter=150)
    end = time.time()
    sec = end - start
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60

    # Guardar Resultados en un archivo
    output_filename = "JD-Results(12)_optimized.txt"
    with open(output_filename, "w") as f:
        f.write("# Resultados del barrido de la grilla en (mu, zt/U) (versión optimizada)\n")
        f.write("# Columnas: mu, zt/U, psi_o, psi_e, rho_o, rho_e, psi_total, delta_rho\n")
        f.write("# Metadatos:\n")
        f.write(f"# base_params: {base_params}\n")
        f.write(f"# dim: {dim}\n")
        f.write(f"# psi_vals_o: {psi_vals_o}\n")
        f.write(f"# psi_vals_e: {psi_vals_e}\n")
        f.write(f"# rho_vals: {rho_vals}\n")
        f.write(f"# Tiempo de ejecución: {int(hours)}:{int(mins)}:{sec:.2f}\n")
        f.write("#\n")
        f.write("#mu\tzt/U\tpsi_o\tpsi_e\trho_o\trho_e\tpsi_total\tdelta_rho\n")
        for row in data_rows:
            f.write("\t".join([f"{x:.6f}" for x in row]) + "\n")
    
    print(f"Archivo de resultados guardado en: {output_filename}")
    print(f"# Tiempo de ejecución: {int(hours)}:{int(mins)}:{sec:.2f}")
