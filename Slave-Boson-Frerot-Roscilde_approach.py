import numpy as np
import math as ma
from scipy.sparse import lil_matrix
from scipy.linalg import eig, inv
import matplotlib.pyplot as plt


def main():
  h_field = 1e-4  
  # Constantes
  f = 6
  U = 1
  mu = np.sqrt(2) - 1
  d = 2
  z = 2 * d   # Nota: en el paper se indica z=6 para d=3; aquí se usa tu definición
  L_A = 50

  CORTE = 50
  step = 0.01

  # Definición de Operadores
  def Ham(phi):
    # Se añade el término fuente -h_field*(b+b†) para regularizar el modo cero
    return ( -(b_op.todense() + b_op_tr.todense()) * z * phi 
             + I.todense() * z * (phi**2) 
             + (U/(2*t0)) * (n_sq - n_op.todense()) 
             - (mu/t0) * n_op.todense() 
             - h_field*(b_op.todense() + b_op_tr.todense()) )

  def F_ab(alpha, beta, asterisco):
    psi_alpha = H_eigsh[1][:, alpha].reshape(f+1, 1)
    psi_beta = H_eigsh[1][:, beta].reshape(f+1, 1)
    psi_alpha = psi_alpha.reshape(1, f+1)
    if not asterisco:
      return psi_alpha * b_op_tr.todense() * psi_beta
    else:
      return psi_beta.reshape(1, f+1) * b_op.todense() * psi_alpha.reshape(f+1, 1)
    
  def EE(vn):
    return np.sum((1 + vn) * np.log(1 + vn) - vn * np.log(vn + 1e-5))
    
  # Operador número y otros operadores base
  n_op = lil_matrix((f+1, f+1))
  for i in range(f+1):
    n_op[i, i] = i

  b_op = lil_matrix((f+1, f+1))
  for i in range(f+1):
    if i > 0:
      b_op[i, i-1] = np.sqrt(i)
  b_op_tr = b_op.transpose()

  I = lil_matrix((f+1, f+1))
  for i in range(f+1):
    I[i, i] = 1

  n_sq = np.matmul(n_op.todense(), n_op.todense())

  # Arrays para almacenamiento de resultados
  ratio = []
  EE_array = []
  wa1 = []
  wa2 = []
  wa3 = []
  wa4 = []
  wa5 = []
  wa6 = []

  # Iteración en t0 (campo medio)
  for i_step in range(CORTE):
    t0 = (0.001 + step * i_step) / z
    ratio.append(t0 * z / U)
    print('\n current: ', i_step, '/', CORTE - 1, '     t0=', t0 * z / U)

    # Solución autoconsistente de campo medio
    err = 1
    phi = 0.1
    while err > ma.pow(10, -6):
      H = Ham(phi)
      w, v = np.linalg.eigh(H)
      idx = np.argsort(w)
      w = w[idx]
      v = v[:, idx]
      H_eigsh = (w, v)
      GS = v[:, 0].reshape(f+1, 1)
      GS_T = GS.reshape(1, f+1)
      E_0 = w[0]
      phi_new = (GS_T * b_op.todense() * GS)[0, 0]
      err = abs(phi_new - phi)
      phi = phi_new

    # Cálculo de la EE sobre el subespacio k
    ee = 0
    wa1_n = 0
    wa2_n = 0
    wa3_n = 0
    wa4_n = 0
    wa5_n = 0
    wa6_n = 0
    for i_k in range(int(L_A) + 1):
      ky = (2 * ma.pi / L_A) * ((-L_A/2) + i_k)
      kx = 0  # kx fijado en 0
      eta_k = (ma.cos(ky) + ma.cos(kx)) / d

      # Construcción del Hamiltoniano cuadrático para este k
      A_0 = lil_matrix((f, f))
      for i in range(f):
        A_0[i, i] = (H_eigsh[0][i+1] - E_0) * t0

      A_1 = lil_matrix((f, f))
      for i in range(f):
        for j in range(f):
          A_1[i, j] = -t0 * (F_ab(i+1, 0, False) * F_ab(j+1, 0, True) + 
                              F_ab(0, j+1, False) * F_ab(0, i+1, True))

      B = lil_matrix((f, f))
      for i in range(f):
        for j in range(f):
          B[i, j] = -t0 * (F_ab(0, i+1, False) * F_ab(j+1, 0, True) + 
                            F_ab(0, j+1, False) * F_ab(i+1, 0, True))

      A_k = A_0 + z * eta_k * A_1
      B_k = z * eta_k * B

      Matrix_AB = lil_matrix((2*f, 2*f))
      for i in range(2*f):
        for j in range(2*f):
          if i < f and j < f:
            Matrix_AB[i, j] = A_k[i, j]
          elif i < f and j >= f:
            Matrix_AB[i, j] = B_k[i, j - f]
          elif i >= f and j < f:
            Matrix_AB[i, j] = B_k[j, i - f]
          else:  # i>=f and j>=f
            Matrix_AB[i, j] = A_k[j - f, i - f]

      Y = np.block([[np.eye(f), np.zeros((f, f))], [np.zeros((f, f)), -np.eye(f)]])
      MM = np.dot(Y, Matrix_AB.todense())
      Qo = np.linalg.eig(MM)[1]
      sp = np.argsort(np.diag(np.dot(np.linalg.inv(Qo), np.dot(Y, np.dot(Matrix_AB.todense(), Qo)))))
      lv = np.concatenate((sp[f:], sp[:f][::-1]))
      QQBO = Qo[:, lv]
      vn = (np.sort(np.linalg.eig(np.dot(Y, np.dot(np.dot(QQBO, np.block([[np.eye(f), np.zeros((f, f))], 
                                                                           [np.zeros((f, f)), np.zeros((f, f))]])), QQBO.T)))[0])[f:2*f] - 1)

      wa1_n += np.log(1 + 1 / (abs(vn[0]) + 1e-8))
      wa2_n += np.log(1 + 1 / (abs(vn[1]) + 1e-8))
      wa3_n += np.log(1 + 1 / (abs(vn[2]) + 1e-8))
      wa4_n += np.log(1 + 1 / (abs(vn[3]) + 1e-8))
      wa5_n += np.log(1 + 1 / (abs(vn[4]) + 1e-8))
      wa6_n += np.log(1 + 1 / (abs(vn[5]) + 1e-8))
      ee += EE(abs(vn))
    
    wa1.append(wa1_n)
    wa2.append(wa2_n)
    wa3.append(wa3_n)
    wa4.append(wa4_n)
    wa5.append(wa5_n)
    wa6.append(wa6_n)
    EE_array.append(ee/L_A)

  # Graficar resultados de la entropía en función de t0
  plt.figure(figsize=(10, 10))
  plt.plot(ratio, EE_array, 'b', label='$S$', marker='.', linestyle='-')
  plt.xlabel('$t_0 z /U$', fontsize=25)
  plt.ylabel('$S_{ent}$', fontsize=35)
  plt.yticks(fontsize=13)
  plt.xticks(fontsize=13)
  plt.legend(fontsize=13)
  plt.grid()
  plt.text(0.08, 0.9, '$L_A=$' + str(L_A), transform=plt.gca().transAxes, fontsize=20,
           bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
  if mu != np.sqrt(2) - 1:
    plt.text(0.08, 0.80, '$\mu=$' + str(mu), transform=plt.gca().transAxes, fontsize=20,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
  else:
    plt.text(0.08, 0.80, '$\mu=\sqrt{2}-1$', transform=plt.gca().transAxes, fontsize=20,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
  plt.show()

  # --- CÁLCULO DE LOS ESPECTROS PARA t0 FIXED ---
  # --- CÁLCULO DE LOS ESPECTROS PARA t0 FIXED ---
  t0_fixed = 0.15 / z
  mu = np.sqrt(2) - 1
  t0 = t0_fixed  # Actualizamos t0 para usarlo en Ham

  # Iteración de campo medio para t0_fixed (incluye ahora el término fuente)
  err = 1
  phi = 0.1
  while err > ma.pow(10, -6):
    H = Ham(phi)
    w, v = np.linalg.eigh(H)
    idx = np.argsort(w)
    w = w[idx]
    v = v[:, idx]
    H_eigsh = (w, v)
    GS = v[:, 0].reshape(f+1, 1)
    GS_T = GS.reshape(1, f+1)
    E_0 = w[0]
    phi_new = (GS_T * b_op.todense() * GS)[0, 0]
    err = abs(phi_new - phi)
    phi = phi_new

  # Cálculo del espectro en función de ky con el Hamiltoniano actualizado
  ky_values = []
  wa1_spec = []
  wa2_spec = []

  for i_k in range(int(L_A) + 1):
    ky = (2 * ma.pi / L_A) * ((-L_A/2) + i_k)
    ky_values.append(ky)
        
    kx = 0  # kx fijado en 0
    eta_k = (ma.cos(ky) + ma.cos(kx)) / d

    A_0 = lil_matrix((f, f))
    for i in range(f):
      A_0[i, i] = (H_eigsh[0][i+1] - E_0) * t0_fixed

    A_1 = lil_matrix((f, f))
    for i in range(f):
      for j in range(f):
        A_1[i, j] = -t0_fixed * (F_ab(i+1, 0, False) * F_ab(j+1, 0, True) +
                                 F_ab(0, j+1, False) * F_ab(0, i+1, True))

    B = lil_matrix((f, f))
    for i in range(f):
      for j in range(f):
        B[i, j] = -t0_fixed * (F_ab(0, i+1, False) * F_ab(j+1, 0, True) +
                               F_ab(0, j+1, False) * F_ab(i+1, 0, True))

    A_k = A_0 + z * eta_k * A_1
    B_k = z * eta_k * B

    Matrix_AB = lil_matrix((2*f, 2*f))
    for i in range(2*f):
      for j in range(2*f):
        if i < f and j < f:
          Matrix_AB[i, j] = A_k[i, j]
        elif i < f and j >= f:
          Matrix_AB[i, j] = B_k[i, j - f]
        elif i >= f and j < f:
          Matrix_AB[i, j] = B_k[j, i - f]
        else:
          Matrix_AB[i, j] = A_k[j - f, i - f]

    Y = np.block([[np.eye(f), np.zeros((f, f))],
                  [np.zeros((f, f)), -np.eye(f)]])
    MM = np.dot(Y, Matrix_AB.todense())
    Qo = np.linalg.eig(MM)[1]
    sp = np.argsort(np.diag(np.dot(np.linalg.inv(Qo), np.dot(Y, np.dot(Matrix_AB.todense(), Qo)))))
    lv = np.concatenate((sp[f:], sp[:f][::-1]))
    QQBO = Qo[:, lv]

    MP = np.block([[np.eye(f), np.zeros((f, f))],
                   [np.zeros((f, f)), np.zeros((f, f))]])
    CC = np.dot(np.dot(QQBO, MP), QQBO.transpose())
    lc, qc = np.linalg.eig(np.dot(Y, CC))
    lc = np.real(lc)  # Para mayor estabilidad numérica
    sorted_indices = np.argsort(lc)
    vn_all = lc[sorted_indices] - 1
    # Extraemos los dos primeros valores (se espera que sean distintos tras incluir h_field)
    vn1 = vn_all[f]   # primer modo del subgrupo positivo
    vn2 = vn_all[f+1] # segundo modo del subgrupo positivo
    wa1_k = np.log(1 + 1 / (abs(vn1) + 1e-5))
    wa2_k = np.log(1 + 1 / (abs(vn2) + 1e-5))
        
    wa1_spec.append(wa1_k)
    wa2_spec.append(wa2_k)

  plt.figure(figsize=(10, 6))
  plt.plot(ky_values, wa1_spec, marker='o', label="wa1")
  plt.plot(ky_values, wa2_spec, marker='s', label="wa2")
  plt.xlabel("$k_y$", fontsize=14)
  plt.ylabel("Valor del espectro", fontsize=14)
  plt.title("Espectros wa1 y wa2 vs $k_y$", fontsize=16)
  plt.legend(fontsize=12)
  plt.grid(True)
  plt.show()


if __name__ == "__main__":
    main()
