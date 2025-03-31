import numpy as np
import time
from math import factorial, sqrt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# ---------------------------- FUNCIONES AUXILIARES ----------------------------
def partial_time(t_start, t_end):
    sec = t_end - t_start
    mins = sec // 60
    sec = sec % 60
    hour = mins // 60
    mins = mins % 60
    return hour, mins, sec

def generate_basis(num_sites, total_particles):
    if num_sites == 1:
        return [(total_particles,)]
    else:
        basis_list = []
        for n in range(total_particles + 1):
            for tail in generate_basis(num_sites - 1, total_particles - n):
                basis_list.append((n,) + tail)
        return basis_list

def compute_label(state):
    label = 0.0
    for i, n in enumerate(state):
        p_i = 100 * (i + 1) + 3
        label += sqrt(p_i) * n
    return round(label, 12)

def generate_reduced_basis(num_sites, max_particles):
    reduced_basis = []
    for n in range(max_particles + 1):
        reduced_basis.extend(generate_basis(num_sites, n))
    return sorted(reduced_basis, reverse=True)

def compute_expectation(psi, basis, site, power=1):
    exp_val = 0.0
    for idx, state in enumerate(basis):
        n_val = state[site]
        exp_val += (abs(psi[idx])**2) * (n_val**power)
    return exp_val

def compute_bdag_b_expectation(psi, basis, i, j, compute_label, label_to_index):
    exp_val = 0.0
    for idx, state in enumerate(basis):
        if state[j] > 0:
            new_state = list(state)
            new_state[i] += 1
            new_state[j] -= 1
            amp_factor = sqrt((state[i] + 1) * state[j])
            lbl_new = compute_label(tuple(new_state))
            new_index = label_to_index.get(lbl_new, None)
            if new_index is not None:
                exp_val += psi[new_index] * psi[idx] * amp_factor
    return exp_val

def compute_reduced_density_matrix(psi, full_basis, A_indices, B_indices):
    grupos = {}
    for idx, state in enumerate(full_basis):
        a_state = tuple(state[k] for k in A_indices)
        b_state = tuple(state[k] for k in B_indices)
        if b_state not in grupos:
            grupos[b_state] = {}
        if a_state not in grupos[b_state]:
            grupos[b_state][a_state] = 0.0 + 0.0j
        grupos[b_state][a_state] += psi[idx]
    
    reduced_basis_A = sorted({a_state for grupo in grupos.values() for a_state in grupo.keys()}, reverse=True)
    A_state_to_index = {state: i for i, state in enumerate(reduced_basis_A)}
    dim_A = len(reduced_basis_A)
    rho_A = np.zeros((dim_A, dim_A), dtype=complex)
    
    for grupo in grupos.values():
        keys = list(grupo.keys())
        for i in range(len(keys)):
            a_state_i = keys[i]
            psi_sum_i = grupo[a_state_i]
            i_idx = A_state_to_index[a_state_i]
            # Contribución diagonal
            rho_A[i_idx, i_idx] += psi_sum_i * np.conjugate(psi_sum_i)
            # Solo para i < j, aprovechando la simetría
            for j in range(i + 1, len(keys)):
                a_state_j = keys[j]
                psi_sum_j = grupo[a_state_j]
                j_idx = A_state_to_index[a_state_j]
                contrib = psi_sum_i * np.conjugate(psi_sum_j)
                rho_A[i_idx, j_idx] += contrib
                rho_A[j_idx, i_idx] += np.conjugate(contrib)

    
    return rho_A, reduced_basis_A

def entanglement_entropy(rho_A):
    evals, _ = np.linalg.eigh(rho_A)
    S = -sum(ev * np.log2(ev) for ev in evals if ev > 1e-12)
    return S


# ---------------------------- CONSTRUCCIÓN DEL HAMILTONIANO ----------------------------
def build_hamiltonian(basis, label_to_index, t, U, mu, M, g_eff, J_B, J_D, epsilon_pert):
    D = len(basis)
    # Hamiltoniano de Bose-Hubbard (términos locales e hopping)
    H_b = lil_matrix((D, D), dtype=float)
    for idx, state in enumerate(basis):
        diag_term = 0.5 * sum(n * (n - 1) for n in state)
        chem_term = -mu * sum(state)
        H_b[idx, idx] = U * diag_term + chem_term
    for idx, state in enumerate(basis):
        state_array = list(state)
        for i in range(M):
            j = (i + 1) % M
            # Término de hopping: b_i† b_j
            if state_array[j] > 0:
                new_state = state_array.copy()
                new_state[i] += 1
                new_state[j] -= 1
                amp = -t * sqrt((state_array[i] + 1) * state_array[j])
                lbl_new = compute_label(tuple(new_state))
                new_index = label_to_index.get(lbl_new, None)
                if new_index is not None:
                    H_b[new_index, idx] += amp
            # Término de hopping: b_j† b_i
            if state_array[i] > 0:
                new_state = state_array.copy()
                new_state[j] += 1
                new_state[i] -= 1
                amp = -t * sqrt((state_array[j] + 1) * state_array[i])
                lbl_new = compute_label(tuple(new_state))
                new_index = label_to_index.get(lbl_new, None)
                if new_index is not None:
                    H_b[new_index, idx] += amp
    H_b = H_b.tocsr()

    # Operador de enlace B:
    B_op = lil_matrix((D, D), dtype=float)
    for idx, state in enumerate(basis):
        state_array = list(state)
        for i in range(M):
            j = (i + 1) % M
            sign = (-1)**i
            # b_i† b_{i+1}
            if state_array[j] > 0:
                new_state = state_array.copy()
                new_state[i] += 1
                new_state[j] -= 1
                amp = sign * sqrt((state_array[i] + 1) * state_array[j])
                lbl_new = compute_label(tuple(new_state))
                new_index = label_to_index.get(lbl_new, None)
                if new_index is not None:
                    B_op[new_index, idx] += amp
            # b_{i+1}† b_i
            if state_array[i] > 0:
                new_state = state_array.copy()
                new_state[i] -= 1
                new_state[j] += 1
                amp = sign * sqrt((state_array[j] + 1) * state_array[i])
                lbl_new = compute_label(tuple(new_state))
                new_index = label_to_index.get(lbl_new, None)
                if new_index is not None:
                    B_op[new_index, idx] += amp
    B_op = B_op.tocsr()

    # Operador de densidad D (con signo alternante)
    D_op = lil_matrix((D, D), dtype=float)
    for idx, state in enumerate(basis):
        D_val = sum(((-1)**i) * n for i, n in enumerate(state))
        D_op[idx, idx] = D_val
    D_op = D_op.tocsr()

    # Cálculo de los cuadrados y términos cruzados
    B2 = B_op.dot(B_op)
    D2 = D_op.dot(D_op)
    BD = B_op.dot(D_op)
    DB = D_op.dot(B_op)

    # Término extra del Hamiltoniano:
    # H_extra = (g_eff/M)[J_B^2 * B^2 + J_D^2 * D^2 + J_D J_B (B D + D B)]
    H_extra = (g_eff/M) * ((J_B**2)*B2 + (J_D**2)*D2 + (J_D*J_B)*(BD + DB))
    
    # Término perturbativo: H_pert = epsilon_pert * D_op
    H_pert = epsilon_pert * D_op

    # Hamiltoniano efectivo completo
    H_eff = H_b + H_extra + H_pert
    return H_eff

# ---------------------------- SIMULACIÓN ----------------------------
def simulate_t(t_val, basis, label_to_index, A_half, B_half, A_even, B_even, N, M, U, mu, t_start, g_eff, J_B, J_D, epsilon_pert):
    t_val_actual = U * t_val
    H_csr = build_hamiltonian(basis, label_to_index, t_val_actual, U, mu, M, g_eff, J_B, J_D, epsilon_pert)
    eigval, eigvec = eigsh(H_csr, k=1, which='SA')
    psi_ground = eigvec[:, 0] / np.linalg.norm(eigvec[:, 0])
    
    # Cálculo de expectativas y entropía
    site1 = 0
    site2 = 1
    n1 = compute_expectation(psi_ground, basis, site1, power=1)
    n1_2 = compute_expectation(psi_ground, basis, site1, power=2)
    var_n1 = n1_2 - n1**2
    n2 = compute_expectation(psi_ground, basis, site2, power=1)
    n2_2 = compute_expectation(psi_ground, basis, site2, power=2)
    var_n2 = n2_2 - n2**2

    superfluid_sum = 0.0
    for i in range(M):
        j = (i + 1) % M
        term = (compute_bdag_b_expectation(psi_ground, basis, i, j, compute_label, label_to_index) +
                compute_bdag_b_expectation(psi_ground, basis, j, i, compute_label, label_to_index))
        superfluid_sum += term
    sf_factor = superfluid_sum / (2 * M)

    rho_A_half, _ = compute_reduced_density_matrix(psi_ground, basis, A_half, B_half)
    S_half = entanglement_entropy(rho_A_half)

    rho_A_even, _ = compute_reduced_density_matrix(psi_ground, basis, A_even, B_even)
    S_even = entanglement_entropy(rho_A_even)

    # ------------------- CÁLCULO DE LOS PARÁMETROS DE ORDEN -------------------
    # Reconstrucción de los operadores B y D para el cálculo de <B^2> y <D^2>
    D_dim = len(basis)
    # Operador de enlace B:
    B_op = lil_matrix((D_dim, D_dim), dtype=float)
    for idx, state in enumerate(basis):
        state_array = list(state)
        for i in range(M):
            j = (i + 1) % M
            sign = (-1)**i
            if state_array[j] > 0:
                new_state = state_array.copy()
                new_state[i] += 1
                new_state[j] -= 1
                amp = sign * sqrt((state_array[i] + 1) * state_array[j])
                lbl_new = compute_label(tuple(new_state))
                new_index = label_to_index.get(lbl_new, None)
                if new_index is not None:
                    B_op[new_index, idx] += amp
            if state_array[i] > 0:
                new_state = state_array.copy()
                new_state[i] -= 1
                new_state[j] += 1
                amp = sign * sqrt((state_array[j] + 1) * state_array[i])
                lbl_new = compute_label(tuple(new_state))
                new_index = label_to_index.get(lbl_new, None)
                if new_index is not None:
                    B_op[new_index, idx] += amp
    B_op = B_op.tocsr()
    
    # Operador de densidad D (con signo alternante)
    D_op = lil_matrix((D_dim, D_dim), dtype=float)
    for idx, state in enumerate(basis):
        D_val = sum(((-1)**i) * n for i, n in enumerate(state))
        D_op[idx, idx] = D_val
    D_op = D_op.tocsr()
    
    B2 = B_op.dot(B_op)
    D2 = D_op.dot(D_op)
    
    exp_B2 = np.real(psi_ground.conj().dot(B2.dot(psi_ground)))
    exp_D2 = np.real(psi_ground.conj().dot(D2.dot(psi_ground)))
    
    O_B = np.sqrt(exp_B2 / (N**2))
    O_DW = np.sqrt(exp_D2 / (N**2))
    
    # Se retornan además los nuevos parámetros de orden
    return [t_val, n1, var_n1, n2, var_n2, sf_factor, S_half, S_even, O_B, O_DW]

# ---------------------------- MAIN ----------------------------
def main():
    t_start = time.time()
    
    # Parámetros del modelo
    N = 8          # Número de partículas
    M = 8          # Número de sitios
    U = 1.0         # Amplitud de interacción local
    mu = np.sqrt(2)-1        # Potencial químico
    # Nuevos parámetros para términos extra y perturbativos
    g_eff = -1.0
    J_B = 0.5
    J_D = 2.0
    epsilon_pert = 1e-4*0  # Término perturbativo para romper la degeneración en MI
    num_steps = 35   # Número de puntos a calcular
    t_over_U_vals = np.logspace(-3, 2.5, num=num_steps)
    
    # Pre-cálculos: base de Fock y mapeo de etiquetas
    all_states = generate_basis(M, N)
    basis = sorted(all_states, reverse=True)
    D = len(basis)
    label_to_index = {}
    for idx, state in enumerate(basis):
        lbl = compute_label(state)
        label_to_index[lbl] = idx

    # Definir particiones:
    # half-half
    A_half = list(range(M // 2))
    B_half = list(range(M // 2, M))
    # even-odd
    A_even = list(range(0, M, 2))
    B_even = list(range(1, M, 2))

    # Función parcial para simulación paralela
    simulate_func = partial(simulate_t, basis=basis, label_to_index=label_to_index,
                            A_half=A_half, B_half=B_half, A_even=A_even, B_even=B_even,
                            N=N, M=M, U=U, mu=mu, t_start=t_start,
                            g_eff=g_eff, J_B=J_B, J_D=J_D, epsilon_pert=epsilon_pert)
    
    with ProcessPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(simulate_func, t_over_U_vals))
    
    results = np.array(results)
    t_end = time.time()
    hour, mins, sec = partial_time(t_start, t_end)
    
    # Guardar resultados (se añaden los nuevos parámetros de orden)
    metadata = (f"# N = {N}\n# M = {M}\n# D = {D}\n# mu = {mu}\n"
                f"# g_eff = {g_eff}\n# J_B = {J_B}\n# J_D = {J_D}\n"
                f"# epsilon_pert = {epsilon_pert}\n# Tiempo de cómputo = {hour}:{mins}:{sec}\n")
    header = "t_U\tn1\tvar_n1\tn2\tvar_n2\tsf_factor\tS_half\tS_even\tO_B\tO_DW"
    output_filename = "resultsJBJD8.txt"
    with open(output_filename, "w") as f:
        f.write(metadata)
        f.write(header + "\n")
        np.savetxt(f, results, fmt="%.6e", delimiter="\t")
    
    print("Resultados guardados en", output_filename)
    print(f"Tiempo total de cómputo: {hour}:{mins}:{sec}")

if __name__ == '__main__':
    main()
