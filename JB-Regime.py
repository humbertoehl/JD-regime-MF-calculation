import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Parámetros globales (ajusta según tu caso)
Ns     = 100           # número total de sitios
z      = 6             # coordinancia de la red
J_B    = 0.0          # coeficiente de bond coupling JB
U      = 1.0           # interacción in situ U
g_eff  = -0.25        # acoplo efecto geff
n_max  = 5             # truncación del espacio de ocupación
res=25

# Creación de operadores en espacio truncado
def create_annihilation(n_max):
    a = np.zeros((n_max, n_max), complex)
    for n in range(1, n_max):
        a[n-1, n] = np.sqrt(n)
    return a

a_op    = create_annihilation(n_max)
adag_op = a_op.T.conj()
n_op    = np.diag(np.arange(n_max))
Id      = np.eye(n_max)
# operador n(n-1):
n_n_minus_one = np.diag([n*(n-1) for n in range(n_max)])

# vecinos con periodicidad mod 4
def next_site(xi): return (xi + 1) % 4
def prev_site(xi): return (xi - 1) % 4


def beta_op(xi, psi):
    ξp = next_site(xi)
    term = (psi[xi].conj() * psi[ξp]*Id  # placeholder para tipología
            + psi[ξp].conj() * a_op
            + psi[ξp]       * adag_op
            + psi[xi]       * psi[ξp].conj()*Id)
    # restar (ψ*ξ ψξ+1 + c.c.)*Id
    corr = (psi[xi].conj()*psi[ξp] + np.conj(psi[xi].conj()*psi[ξp])) * Id
    return term - corr

def eta_tilde(psi):
    suma = 0
    for xj in range(4):
        xp = next_site(xj)
        suma += ((-1)**(xj+1)) * (psi[xj].conj()*psi[xp] + psi[xp].conj()*psi[xj])
    return (z/8) * suma

def c_tilde(xi, psi, eta):

    ξp = next_site(xi)
    pref = (psi[xi].conj()*psi[ξp] + psi[ξp].conj()*psi[xi])
    return (z/4) * pref * eta

def delta_S2_op(xi, psi, rho):
    # vecinos con periodicidad
    xi_p = (xi + 1) % 4
    xi_m = (xi - 1) % 4

    # coeficiente global z/4
    coeff = (z / 4.0)

    # correlaciones de psi:  ⟨b†_ξ b_{ξ'} + H.c.⟩ = psi[ξ].conj()*psi[ξ'] + c.c.
    corr_p = psi[xi].conj()*psi[xi_p] + (psi[xi].conj()*psi[xi_p]).conj()
    corr_m = psi[xi].conj()*psi[xi_m] + (psi[xi].conj()*psi[xi_m]).conj()

    # términos de dos-particle-hole excitations con vecinos: b†²_ξ b²_{ξ±1} + H.c.
    # decoupled: ⟨b²_{ξ±1}⟩ b†²_ξ + ⟨b†²_{ξ±1}⟩ b²_ξ
    t_pp = (psi[xi_p]**2) * (adag_op @ adag_op) + (psi[xi_p].conj()**2) * (a_op @ a_op)
    t_mm = (psi[xi_m]**2) * (adag_op @ adag_op) + (psi[xi_m].conj()**2) * (a_op @ a_op)

    # términos de densidad local y vecino: 2 n̂_ξ ⟨n̂_{ξ±1}⟩ + ⟨n̂_{ξ±1}⟩ + n̂_ξ
    d_pp = 2.0 * rho[xi_p] * n_op + rho[xi_p] * Id + n_op
    d_mm = 2.0 * rho[xi_m] * n_op + rho[xi_m] * Id + n_op

    # término de corrección vía beta_op
    beta = beta_op(xi, psi)
    term_p = -2.0 * corr_p * beta + (corr_p**2) * Id
    term_m = -2.0 * corr_m * beta + (corr_m**2) * Id

    # ensamblar δS²
    delta_S2 = coeff * (t_pp + t_mm + d_pp + d_mm + term_p + term_m)
    return delta_S2

def build_local_hamiltonian(xi, mu, t0, psi, rho):
    eta  = eta_tilde(psi)
    cB   = c_tilde(xi, psi, eta)
    tphi = -t0 - g_eff*Ns*(J_B**2)*eta
    H_hop  = (z/2)*tphi * beta_op(xi, psi)
    H_mu   = - mu * n_op
    H_U    = (U/2) * n_n_minus_one
    H_qol  = g_eff*(J_B**2) * delta_S2_op(xi, psi, rho)
    H_const= - g_eff*Ns*(J_B**2) * cB * Id
    Hξ = (H_hop + H_mu + H_U + H_qol + H_const) * (Ns/4)
    return Hξ


def fixed_point_4sites(mu, t0, psi_init, rho_init, max_iters=200, tol=1e-3):
    psi = psi_init.copy()
    rho = rho_init.copy()
    Hs = None

    for _ in range(max_iters):
        # Construir y diagonalizar Hξ para ξ=0…3
        Hs = [build_local_hamiltonian(xi, mu, t0, psi, rho) for xi in range(4)]
        eig_results = [eigh(H) for H in Hs]
        ground_states = [vecs[:,0] for (_, vecs) in eig_results]

        # Calcular nuevos ψξ y ρξ
        psi_new = np.array([gs.conj().T @ (a_op @ gs) for gs in ground_states])
        rho_new = np.array([gs.conj().T @ (n_op @ gs) for gs in ground_states])

        # Verificar convergencia
        if np.max(np.abs(psi_new - psi)) < tol and np.max(np.abs(rho_new - rho)) < tol:
            psi, rho = psi_new, rho_new
            break

        psi, rho = psi_new, rho_new

    # Cálculo de energía total en el estado base
    energy = 0.0
    for H, gs in zip(Hs, ground_states):
        energy += np.real(gs.conj().T @ (H @ gs))

    return psi, rho, energy

# Barrido de (μ, t0) y cálculo de ordenes
mu_vals = np.linspace(0, 2.5, res)
t0_vals = np.linspace(0, .25/z, res)
# Número de condiciones iniciales para explorar
N_cond = 30

# Inicializar matrices de resultados
Psi_tot   = np.zeros((len(mu_vals), len(t0_vals)), dtype=float)
Delta_rho = np.zeros_like(Psi_tot)
Rho_tot   = np.zeros_like(Psi_tot)
Delta_phi = np.zeros_like(Psi_tot)

# Barrido de (mu, t0)
for i, mu in enumerate(mu_vals):
    for j, t0 in enumerate(t0_vals):
        min_energy      = np.inf
        best_total_psi  = 0.0
        best_delta_rho  = 0.0
        best_rho_total  = 0.0
        best_delta_phi  = 0.0

        # Probar varias condiciones iniciales
        for k in range(N_cond):
            # Condiciones iniciales aleatorias complejas para psi
            psi_init = np.array([
                np.random.uniform(0, 0.3) * np.random.choice([-1,1])
                for _ in range(4)
            ])
            # Densidades iniciales aleatorias
            rho_init = np.random.uniform(0, 1.4, size=4)

            # Cálculo auto-consistente, incluyendo energía final
            psi, rho, energy = fixed_point_4sites(mu, t0, psi_init, rho_init)

            # Seleccionar la configuración de mínima energía
            if energy < min_energy:
                min_energy      = energy
                best_total_psi  = np.mean(np.abs(psi))
                best_delta_rho  = (rho[0] + rho[1] - rho[2] - rho[3]) / 4.0
                best_rho_total  = np.mean(rho)
                best_delta_phi  = abs(
                    np.angle(psi[0]) + np.angle(psi[1])
                    - np.angle(psi[2]) - np.angle(psi[3])
                )

        # Guardar valores óptimos en las matrices
        Psi_tot[j,i]   = best_total_psi
        Delta_rho[j,i] = best_delta_rho
        Rho_tot[j,i]   = best_rho_total
        Delta_phi[j,i] = best_delta_phi

        # Salida de estado para seguimiento
        print(f"(mu={mu:.3f}, t0={t0:.3f}) -> Psi_tot={best_total_psi:.3f}, Delta_rho={best_delta_rho:.3f}, Rho_tot={best_rho_total:.3f}, Delta_phi={best_delta_phi:.3f}")



fig, axs = plt.subplots(2,2, figsize=(10,8))
c1 = axs[0,0].pcolormesh(t0_vals, mu_vals, Psi_tot.T); fig.colorbar(c1, ax=axs[0,0]); axs[0,0].set_title(r"$\psi_{\rm total}$")
c2 = axs[0,1].pcolormesh(t0_vals, mu_vals, Delta_rho.T); fig.colorbar(c2, ax=axs[0,1]); axs[0,1].set_title(r"$\Delta\rho_D$")
c3 = axs[1,0].pcolormesh(t0_vals, mu_vals, Rho_tot.T); fig.colorbar(c3, ax=axs[1,0]); axs[1,0].set_title(r"$\rho_{\rm total}$")
c4 = axs[1,1].pcolormesh(t0_vals, mu_vals, Delta_phi.T); fig.colorbar(c4, ax=axs[1,1]); axs[1,1].set_title(r"$\Delta\phi$")
plt.tight_layout()
plt.show()
