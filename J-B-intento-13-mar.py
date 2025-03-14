import numpy as np
from scipy.linalg import eigh
import time
from multiprocessing import Pool, cpu_count

# --------------------------------------------------------------------
def create_fock_operators_extended(dim):
    b = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):
        b[n-1, n] = np.sqrt(n)
    b_dag = b.T.conj()
    n_op = np.dot(b_dag, b)
    I = np.eye(dim, dtype=complex)
    # Operadores para b^2
    b2 = np.dot(b, b)
    b2_dag = np.dot(b_dag, b_dag)
    return b, b_dag, b2, b2_dag, n_op, I

# --------------------------------------------------------------------
def mean_field_iteration_bond(initial_params, params, b, b_dag, b2, b2_dag, n_op, I, tol=1e-5, max_iter=150):
    # arámetros de condiciones iniciales para las 4 sublattices
    psi = np.array([initial_params['psi1'], initial_params['psi2'],
                    initial_params['psi3'], initial_params['psi4']], dtype=complex)
    rho = np.array([initial_params['rho1'], initial_params['rho2'],
                    initial_params['rho3'], initial_params['rho4']], dtype=float)
                    
    # Parámetros globales
    z = params['z']
    t0 = params['t0']
    mu = params['mu']
    U = params['U']
    N_s = params['N_s']
    g_eff = params['g_eff']
    JB_abs_sq = np.abs(params['J_B'])**2

    # Iteración punto fijo
    for it in range(max_iter):
        psi_old = psi.copy()
        rho_old = rho.copy()
        #término global
        E_eta = 0.0
        for i in range(4):
            ip = (i+1) % 4
            prod = psi[i].conjugate() * psi[ip]
            E_eta += ((-1)**(i+1)) * (prod + prod.conjugate()).real
        # terminos que varian por sub-red
        tilde_eta = np.zeros(4, dtype=float)
        t_phi = np.zeros(4, dtype=float)
        tilde_c = np.zeros(4, dtype=float)
        # Hamiltonianos, Energías y paŕametros por sub-red
        H_list = []
        energies = np.zeros(4, dtype=float)
        new_psi = np.zeros(4, dtype=complex)
        new_rho = np.zeros(4, dtype=float)
        
        for i in range(4):
            # Índices de vecinos
            ip = (i+1) % 4
            im = (i-1) % 4
            # Término eta para sublattice i
            tilde_eta[i] = (z * ((-1)**(i+1)) / 8.0) * E_eta
            t_phi[i] = t0 - g_eff * N_s * JB_abs_sq * tilde_eta[i]
            beta_i = psi[ip].conjugate() * b + psi[ip] * b_dag - \
                     (psi[i].conjugate()*psi[ip] + psi[i]*np.conjugate(psi[ip])) * I #beta en términos de campo efectivo del vecino
            # constante c̃_B para sub-red i
            c_term = (psi[i].conjugate()*psi[ip] + psi[i]*np.conjugate(psi[ip])).real
            tilde_c[i] = (z/4.0) * c_term * tilde_eta[i]
            
            deltaS2 = np.zeros_like(n_op, dtype=complex)
            for s, j in [ (1, ip), (-1, im) ]:
                # b^2 del vecino se reemplaza por psi_j^2 y n por rho_j.
                psi_j_sq = psi[j]**2
                term1 = np.dot(b2_dag, psi_j_sq)    
                term2 = np.dot(np.conjugate(psi_j_sq), b2)  
                term3 = 2 * rho[j] * n_op
                term4 = n_op + rho[j]*I
                # Término que involucra la correlación de fases:
                phase_corr = (psi[i].conjugate()*psi[j] + psi[i]*np.conjugate(psi[j])).real
                term5 = - 2 * phase_corr * beta_i
                term6 = (phase_corr**2) * I
                deltaS2 += term1 + term2 + term3 + term4 + term5 + term6
            deltaS2 = (z/4.0) * deltaS2

            # Hamiltoniano efectivo para sub-red i
            H_i = (N_s/4.0) * ( (z/2.0)*t_phi[i]*beta_i - mu*n_op + 
                                g_eff * JB_abs_sq * deltaS2 +
                                U*(np.dot(n_op, n_op) - n_op) -
                                g_eff * N_s * JB_abs_sq * tilde_c[i] * I )
            H_list.append(H_i)
            
            # Diagonalización
            E_vals, E_vecs = eigh(H_i)
            idx0 = np.argmin(E_vals)
            energy_i = E_vals[idx0]
            energies[i] = energy_i
            psi_state = E_vecs[:, idx0]
            # Actualización de parámetros
            new_psi[i] = np.vdot(psi_state, np.dot(b, psi_state))
            new_rho[i] = np.vdot(psi_state, np.dot(n_op, psi_state)).real
        
        # convergencia
        diff = np.abs(np.concatenate((new_psi - psi, new_rho - rho)))
        if np.all(diff < tol):
            psi = new_psi
            rho = new_rho
            break
        psi = new_psi
        rho = new_rho

    # Diccionario de parámetros finales:
    final_params = {'psi1': psi[0], 'psi2': psi[1], 'psi3': psi[2], 'psi4': psi[3],
                    'rho1': rho[0], 'rho2': rho[1], 'rho3': rho[2], 'rho4': rho[3]}
    total_energy = np.sum(energies)
    energies_dict = {'E1': energies[0], 'E2': energies[1], 'E3': energies[2], 'E4': energies[3],
                     'E_total': total_energy}
    return final_params, energies_dict

# --------------------------------------------------------------------
def run_mean_field_bond(initial_conditions, params, b, b_dag, b2, b2_dag, n_op, I, tol=1e-5, max_iter=150):
    final_params, energies = mean_field_iteration_bond(initial_conditions, params, b, b_dag, b2, b2_dag, n_op, I, tol, max_iter)
    return final_params, energies

# --------------------------------------------------------------------
def compute_grid_point_bond(args):

    # se aplica mean-field para cada punto zt-mu
    mu_val, zt_val, base_params, psi_candidates, rho_candidates, b, b_dag, b2, b2_dag, n_op, I, tol, max_iter = args
    params_local = base_params.copy()
    params_local['mu'] = mu_val
    params_local['t0'] = zt_val / params_local['z']
    
    best_energy = np.inf
    best_solution = None
    best_init = None
    # Se recorren condiciones iniciales
    for psi_init in psi_candidates:
        for rho_init in rho_candidates:
            init_cond = {
                'psi1': psi_init[0],
                'psi2': psi_init[1],
                'psi3': psi_init[2],
                'psi4': psi_init[3],
                'rho1': rho_init[0],
                'rho2': rho_init[1],
                'rho3': rho_init[2],
                'rho4': rho_init[3]
            }
            sol, energy_dict = run_mean_field_bond(init_cond, params_local, b, b_dag, b2, b2_dag, n_op, I, tol, max_iter)
            energy_total = energy_dict['E_total']
            if energy_total < best_energy:
                best_energy = energy_total
                best_solution = sol.copy()
                best_init = init_cond.copy()

    psi_vals = np.array([best_solution['psi1'], best_solution['psi2'], best_solution['psi3'], best_solution['psi4']])
    avg_psi = np.mean(np.abs(psi_vals))
    
    phase_diff = []
    for i in range(4):
        ip = (i+1) % 4
        diff = np.angle(psi_vals[ip]) - np.angle(psi_vals[i])
        phase_diff.append(diff)
    phase_diff = np.array(phase_diff)
    
    print(f'(mu,zt)=({mu_val:.4f},{zt_val:.4f}) -> |SF|_avg={avg_psi:.4f}, fases={phase_diff}, init={best_init}')

    return (mu_val, zt_val, best_solution['psi1'], best_solution['psi2'], best_solution['psi3'], best_solution['psi4'],
            best_solution['rho1'], best_solution['rho2'], best_solution['rho3'], best_solution['rho4'],
            avg_psi, phase_diff)

# --------------------------------------------------------------------
def sweep_phase_diagram_parallel_bond(mu_vals, zt_vals, base_params, b, b_dag, b2, b2_dag, n_op, I,
                                      psi_candidates, rho_candidates, tol=1e-5, max_iter=150):
    args_list = []
    for mu_val in mu_vals:
        for zt_val in zt_vals:
            args_list.append((mu_val, zt_val, base_params, psi_candidates, rho_candidates,
                              b, b_dag, b2, b2_dag, n_op, I, tol, max_iter))
    
    pool = Pool(processes=cpu_count())
    results = pool.map(compute_grid_point_bond, args_list)
    pool.close()
    pool.join()
    
    # Preparar datos para guardar
    data_rows = []
    for res in results:
        (mu_val, zt_val, psi1, psi2, psi3, psi4, rho1, rho2, rho3, rho4, avg_psi, phase_diff) = res
        # CORRECIÓN PA DESPUÉS: GUARDAR CADA FASE POR SEPARADO, NO SOLO EL PROMEDIO
        avg_phase_diff = np.mean(np.abs(phase_diff))
        data_rows.append([mu_val, zt_val, psi1, psi2, psi3, psi4, rho1, rho2, rho3, rho4, avg_psi, avg_phase_diff, phase_diff[0], phase_diff[1], phase_diff[2], phase_diff[3]])
    
    return data_rows

# --------------------------------------------------------------------
if __name__ == '__main__':

    base_params = {
        'U': 1.0,
        'J_B': 1.0,  # Acoplamiento bond
        'N_s': 100,
        'z': 6,
        'mu': 0.5,       # Se actualiza
        't0': 0.05 / 6.0 # Se actualiza
    }
    base_params['g_eff'] = -0.5 * base_params['U'] / base_params['N_s']  
    

    dim = 12
    b, b_dag, b2, b2_dag, n_op, I = create_fock_operators_extended(dim)
    
    # Grilla de parámetros
    N_mu = 5
    N_zt = 5
    mu_vals = np.linspace(0, 4, N_mu)
    zt_vals = np.linspace(0, 1, N_zt)
    
    # Conjuntos de condiciones iniciales
    psi_candidates = [
        [0.101+0j, 0.102+0j, 0.103+0j,     0.104+0j],     
        [0.101+0j, 0.102+0j, 0.103*np.exp(1j*.2),     0.104*np.exp(1j*.2)],  
        [0.101+0j, 0.102+0j, 0.103*np.exp(1j*.5),     0.104*np.exp(1j*.5)],  
        [0.101+0j, 0.102+0j, 0.103*np.exp(1j*.8),     0.104*np.exp(1j*.8)],   

    ]
    rho_candidates = [
        [0.5, 0.5, 0.5, 0.5],
        [0.5, 0.4, 0.5, 0.4],
        [0.4, 0.5, 0.4, 0.5]
    ]
    
    # Calculo en paralelo
    start = time.time()
    data_rows = sweep_phase_diagram_parallel_bond(mu_vals, zt_vals, base_params,
                                                  b, b_dag, b2, b2_dag, n_op, I,
                                                  psi_candidates, rho_candidates,
                                                  tol=1e-5, max_iter=150)
    end = time.time()
    time_of_execution = (end - start) / 60.0
    print(f"Tiempo de ejecución: {time_of_execution:.2f} min")
    
    # Guardar resultados
    output_filename = "JB-Results(7).txt"
    with open(output_filename, "w") as f:
        f.write("# Resultados del barrido en (mu, zt/U) para régimen bond (J_B≠0, J_D=0)\n")
        f.write("# Columnas: mu, zt/U, psi1, psi2, psi3, psi4, rho1, rho2, rho3, rho4, |psi|_avg, avg_phase_diff, phase1, phase2, phase3, phase4\n")
        f.write("# Metadatos:\n")
        f.write(f"# base_params: {base_params}\n")
        f.write(f"# dim: {dim}\n")
        f.write(f"# psi_candidates: {psi_candidates}\n")
        f.write(f"# rho_candidates: {rho_candidates}\n")
        f.write(f"# Tiempo de ejecución: {time_of_execution:.2f} min\n")
        f.write("#\n")
        f.write("#mu\tzt/U\tpsi1\tpsi2\tpsi3\tpsi4\trho1\trho2\trho3\trho4\t|psi|_avg\tavg_phase_diff\tphase1\tphase2\tphase3\tphase4\n")
        for row in data_rows:
            # Para los parámetros complejos tomo el valor absoluto
            psi_vals = [np.abs(row[i]) if i < 6 and i>=2 else row[i] for i in range(2,6)]
            line = "\t".join([f"{x:.6f}" if isinstance(x, float) else str(x) for x in row[:2]] +
                             [f"{val:.6f}" for val in psi_vals] +
                             [f"{x:.6f}" for x in row[6:10]] +
                             [f"{row[10]:.6f}", f"{row[11]:.6f}", f"{row[12]:.6f}", f"{row[13]:.6f}", f"{row[14]:.6f}", f"{row[15]:.6f}"])
            f.write(line + "\n")
    
    print(f"Archivo de resultados guardado en: {output_filename}")
