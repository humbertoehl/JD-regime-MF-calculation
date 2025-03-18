import numpy as np
import time
from math import factorial, sqrt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# ---------------------------- FUNCIONES AUXILIARES ----------------------------

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

def compute_reduced_density_matrix(psi, full_basis, A_indices, B_indices, max_particles):
    reduced_basis_A = generate_reduced_basis(len(A_indices), max_particles)
    A_state_to_index = {state: i for i, state in enumerate(reduced_basis_A)}
    dim_A = len(reduced_basis_A)
    rho_A = np.zeros((dim_A, dim_A), dtype=complex)
    D_full = len(full_basis)
    for i in range(D_full):
        state_i = full_basis[i]
        a_i = tuple(state_i[k] for k in A_indices)
        b_i = tuple(state_i[k] for k in B_indices)
        for j in range(D_full):
            state_j = full_basis[j]
            a_j = tuple(state_j[k] for k in A_indices)
            b_j = tuple(state_j[k] for k in B_indices)
            if b_i == b_j:
                rho_A[A_state_to_index[a_i], A_state_to_index[a_j]] += psi[i] * np.conjugate(psi[j])
    return rho_A, reduced_basis_A

def entanglement_entropy(rho_A):
    evals, V = np.linalg.eigh(rho_A)
    ln_D = np.diag([np.log(ev) if ev > 1e-12 else 0.0 for ev in evals])
    ln_rho_A = V @ ln_D @ V.conj().T
    S = -np.real(np.trace(rho_A @ ln_rho_A))
    return S

def build_hamiltonian(basis, label_to_index, t, U, mu, M):
    D = len(basis)
    H = lil_matrix((D, D), dtype=float)
    # Término diagonal: interacción y potencial químico
    for idx, state in enumerate(basis):
        diag_term = 0.5 * sum(n * (n - 1) for n in state)
        chem_term = -mu * sum(state)
        H[idx, idx] = U * diag_term + chem_term
    # Término de hopping
    for idx, state in enumerate(basis):
        state_array = list(state)
        for i in range(M):
            j = (i + 1) % M
            # b_i† b_j
            if state_array[j] > 0:
                new_state = state_array.copy()
                new_state[i] += 1
                new_state[j] -= 1
                amp = -t * sqrt((state_array[i] + 1) * state_array[j])
                lbl_new = compute_label(tuple(new_state))
                new_index = label_to_index.get(lbl_new, None)
                if new_index is not None:
                    H[new_index, idx] += amp
            # b_j† b_i
            if state_array[i] > 0:
                new_state = state_array.copy()
                new_state[j] += 1
                new_state[i] -= 1
                amp = -t * sqrt((state_array[j] + 1) * state_array[i])
                lbl_new = compute_label(tuple(new_state))
                new_index = label_to_index.get(lbl_new, None)
                if new_index is not None:
                    H[new_index, idx] += amp
    return H.tocsr()

def simulate_t(t_val, basis, label_to_index, A_half, B_half, A_even, B_even, N, M, U, mu):
    t_val_actual = U * t_val  # t real
    H_csr = build_hamiltonian(basis, label_to_index, t_val_actual, U, mu, M)
    eigval, eigvec = eigsh(H_csr, k=1, which='SA')
    psi_ground = eigvec[:, 0] / np.linalg.norm(eigvec[:, 0])
    
    site1 = 0
    site2 = 1
    n1 = compute_expectation(psi_ground, basis, site1, power=1)
    n1_2 = compute_expectation(psi_ground, basis, site1, power=2)
    var_n1 = n1_2 - n1**2
    n2 = compute_expectation(psi_ground, basis, site2, power=1)
    n2_2 = compute_expectation(psi_ground, basis, site2, power=2)
    var_n2 = n2_2 - n2**2
    
    # Factor superfluido
    superfluid_sum = 0.0
    for i in range(M):
        j = (i + 1) % M
        term = (compute_bdag_b_expectation(psi_ground, basis, i, j, compute_label, label_to_index) +
                compute_bdag_b_expectation(psi_ground, basis, j, i, compute_label, label_to_index))
        superfluid_sum += term
    sf_factor = superfluid_sum / (2 * M)
    
    # Entropía de enredamiento para particiones:
    rho_A_half, _ = compute_reduced_density_matrix(psi_ground, basis, A_half, B_half, N)
    S_half = entanglement_entropy(rho_A_half)
    rho_A_even, _ = compute_reduced_density_matrix(psi_ground, basis, A_even, B_even, N)
    S_even = entanglement_entropy(rho_A_even)
    
    return [t_val, n1, var_n1, n2, var_n2, sf_factor, S_half, S_even]

# ---------------------------- MAIN ----------------------------
def main():
    t_start = time.time()
    
    # Parámetros del modelo
    N = 8          # Número de partículas
    M = 8          # Número de sitios
    U = 1.0        # Amplitud local
    mu = 0.0       # Potencial químico
    num_steps = 2 # Número de puntos a calcular
    t_over_U_vals = np.logspace(-2, 2, num=num_steps)
    
    # Pre-cálculos comunes: base de Fock completa y mapeo de etiquetas
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

    
    # función paralela para cada t/U
    simulate_func = partial(simulate_t, basis=basis, label_to_index=label_to_index,
                            A_half=A_half, B_half=B_half, A_even=A_even, B_even=B_even,
                            N=N, M=M, U=U, mu=mu)
    
    # Ejecutar en paralelo 
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(simulate_func, t_over_U_vals))
    
    results = np.array(results)
    t_end = time.time()
    sec = t_end - t_start
    mins = sec // 60
    sec = sec % 60
    hour = mins // 60
    mins = mins % 60
    
    # Guardar resultados
    metadata = f"# N = {N}\n# M = {M}\n# D = {D}\n# mu = {mu}\n# Computation time = {hour}:{mins}:{sec} \n"
    header = "t_U\tn1\tvar_n1\tn2\tvar_n2\tsf_factor\tS_half\tS_even"
    output_filename = "results.txt"
    with open(output_filename, "w") as f:
        f.write(metadata)
        f.write(header + "\n")
        np.savetxt(f, results, fmt="%.6e", delimiter="\t")
    
    print("Resultados guardados en", output_filename)
    print(f"Tiempo total de cómputo: {hour}:{mins}:{sec}")

if __name__ == '__main__':
    main()
