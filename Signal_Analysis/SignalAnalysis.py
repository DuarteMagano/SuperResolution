import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import scipy
from sklearn.linear_model import Lasso
from scipy.linalg import toeplitz
from tqdm import trange
from scipy.optimize import curve_fit, lsq_linear
from functools import partial
import Quantum_Simulation
from IPython.display import clear_output
import os

class Fourier_Transform():
    def __init__(self, t, signal, norm = 1):
        self.t = t
        self.signal = signal
        self.norm = norm

    def get_DFT(self, plot = True, w_exp=[]):
        F_w = np.abs(np.fft.fft(self.signal, self.signal.size))
        F_w /= (np.max(F_w) * self.norm)
        omega = np.fft.fftfreq(self.t.size, self.t[1] - self.t[0]) * 2 * np.pi
        index_sorted = np.argsort(omega)
        omega, F_w = omega[index_sorted], F_w[index_sorted]

        if not plot:
            return omega, F_w
        
        plt.plot(omega, F_w)
        plt.xlabel(r"$\omega$")
        plt.ylabel(r"$|F(\omega)|$")
        if w_exp:
            for w in w_exp:
                plt.axvline(x=w, color="red", ls=":")
                plt.axvline(x=-w, color="red", ls=":")
    
        return omega, F_w



class Compressive_Sensing():
    def __init__(self, m):
        self.m = m
    
    def get_spectrum(self, t, signal, lam):
        n = signal.size
        x = cp.Variable(n, complex=True)
        index = np.random.choice(n, self.m, replace=False)
        A = np.fft.fft(np.eye(n))[index] / np.sqrt(n)
        L = 1/2 * cp.norm2(A @ x - signal[index]) ** 2 + lam * cp.norm1(x)

        problem = cp.Problem(cp.Minimize(L))
        problem.solve(solver=cp.SCS, verbose=False)
        return x.value


    def plot_result(self, t, signal, lam = 1e-6, plot = True, w_exp = []):
        x = self.get_spectrum(t, signal -1/2, lam)
        F_w = np.abs(x)
        F_w /= np.max(F_w)
        omega = np.fft.fftfreq(t.size, t[1] - t[0]) * 2 * np.pi

        if not plot:
            return omega[np.argsort(F_w)][::-1], F_w[np.argsort(F_w)][::-1]

        plt.plot(omega, F_w)
        plt.xlabel(r"$\omega$")
        plt.ylabel(r"$|F(\omega)|$")
        if w_exp:
            for w in w_exp:
                plt.axvline(x=w, color="red", ls=":")
    
        return omega[np.argsort(F_w)][::-1], F_w[np.argsort(F_w)][::-1]


class Atomic_Norm_Minimization():
    def __init__(self, t, z):
        self.t = t
        self.z = z
        self.n = np.size(z)


    def cp_toeplitz(self, u):
        return cp.vstack([
            cp.hstack([u[abs(i - j)] for j in range(self.n)])
            for i in range(self.n)
        ])


    def cp_build_block_matrix_ADMM(self, U, x, t):
        x_col = cp.reshape(x, (self.n, 1), order = 'C')             
        x_row = cp.reshape(cp.conj(x), (1, self.n), order = 'C')  
        t_mat = cp.reshape(t, (1, 1), order = 'C')                        
        return cp.bmat([
            [U,     x_col],
            [x_row, t_mat]
            ])


    def Lagrangian(self, x, u, t, Sigma, S):
        z = cp.Constant(self.z)
        U = self.cp_toeplitz(u)
        L1 = 0.5 * cp.sum_squares(x - z) + self.lamb / 2 * (1 / self.n * cp.trace(U) + t)
        M = self.cp_build_block_matrix_ADMM(U, x, t)
        L2 = cp.trace(Sigma.H @ (S - M))
        L3 = self.rho / 2 * cp.norm(S - M) ** 2
        return cp.real(L1 + L2 + L3)
    

    def atomic_norm_admm(self, lam=1.0, rho=1.0, max_iter=100, tol=1e-4):
        n = self.n
        Sigma = np.zeros((n + 1, n + 1), dtype=complex)
        S = np.zeros((n + 1, n + 1), dtype=complex)

        for iter_num in trange(max_iter, desc=f"ADMM Iterations lambda = {lam} "):
        # === Step 1: Minimize over x, u, t with Toeplitz constraints ===
            x = cp.Variable(n, complex=True)
            u = cp.Variable(n, complex=True)
            t = cp.Variable()
            Toeplitz_u = cp.Variable((n, n), hermitian=True)

        # Toeplitz constraints: Toeplitz_u[i, i+k] = u[k] and hermitian symmetry
            toeplitz_constraints = []
            for k in range(n):
                for i in range(n - k):
                    toeplitz_constraints.append(Toeplitz_u[i, i + k] == u[k])
                    toeplitz_constraints.append(Toeplitz_u[i + k, i] == cp.conj(u[k]))

        # Construct lifted matrix X = [[Toeplitz_u, x], [x^H, t]]
            X_cvx = cp.bmat([
            [Toeplitz_u, cp.reshape(x, (n, 1))],
            [cp.reshape(cp.conj(x), (1, n)), cp.reshape(t, (1, 1))]])

            Delta1 = S - X_cvx

        # ADMM subproblem objective
            obj1 = (
            0.5 * cp.sum_squares(x - self.z)
            + lam / 2 * (cp.trace(Toeplitz_u) / n + t)
            + cp.real(cp.trace(Sigma.conj().T @ Delta1))
            + (rho / 2) * cp.norm(Delta1, 'fro') ** 2)

            prob1 = cp.Problem(cp.Minimize(obj1), toeplitz_constraints)
            prob1.solve(solver=cp.SCS, verbose=False)

            if x.value is None:
                raise ValueError("CVXPY failed to solve the subproblem in Step 1")

        # Extract solution from Step 1
            x_val = x.value
            u_val = u.value
            t_val = t.value
            Toeplitz_val = toeplitz(np.conj(u_val), u_val)  # Hermitian Toeplitz
            X_val = np.block([
            [Toeplitz_val, x_val.reshape(-1, 1)],
            [x_val.conj().reshape(1, -1), np.array([[t_val]])]])

        # === Step 2: PSD projection (replaces full SDP) ===
        # Y = X - (1/rho) * Sigma
            Y = X_val - (1 / rho) * Sigma

        # Hermitian safety check
            Y = (Y + Y.conj().T) / 2

            eigvals, eigvecs = np.linalg.eigh(Y)
            eigvals_clipped = np.clip(eigvals, 0, None)  # project negative eigvals to 0
            S = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.conj().T

        # === Step 3: Dual variable update ===
            Sigma += rho * (S - X_val)

        # === Convergence check ===
            primal_residual = np.linalg.norm(S - X_val, 'fro')
            if primal_residual < tol:
                break

        return x_val
    

    def get_u_atomic_norm_minimization(self, x):
        x = cp.Constant(x)
        u = cp.Variable(self.n, complex = True)
        t = cp.Variable(1)
        T = self.cp_toeplitz(u)
        L = cp.real(1 / (2 * self.n) * cp.trace(T) + (1 / 2) * t)

        problem = cp.Problem(cp.Minimize(L), [self.cp_build_block_matrix_ADMM(T, x, t) >> 0, t >= 0])
        problem.solve(solver = cp.SCS)
        return u.value
    


    def cp_build_block_matrix_Polynomial(self, H, p):
        p_col = cp.reshape(p, (self.n, 1), order = 'C')             
        p_row = cp.conj(cp.transpose(p_col))
        one = cp.reshape(1, (1, 1), order = 'C')        
        return cp.bmat([
            [H,     p_col],
            [p_row, one]
            ])
    

    def get_polynomial(self, x, noise = False, lam = 0):
        x = cp.Constant(x)
        p = cp.Variable(self.n, complex = True)
        H = cp.Variable((self.n, self.n), hermitian = True)
        L = cp.real(cp.conj(x) @ p)
        constraints = [self.cp_build_block_matrix_Polynomial(H, p) >> 0]
        constraints.append(cp.sum(cp.diag(H)) == 1)
        for k in range(1, self.n):
            constraints.append(cp.sum(cp.diag(H, k)) == 0)

        if noise:
            L = L - lam / 2 * cp.norm2(p) ** 2
            #L = cp.real(cp.conj(x - lam * p) @ p) - lam * cp.norm2(p)
            
        problem = cp.Problem(cp.Maximize(L), constraints)
        problem.solve(solver = cp.SCS)
        return p.value
    
    def scale_signal(self, x, T, w_in, w_f):
        dw = (self.n - 1) * np.pi / T - (w_f - w_in) 
        t = np.linspace(-T, T, self.n)
        return (dw, np.exp(- 1j * (w_in - dw / 2) * t) * x)      
    
    def scale_freq(self, nu, w_in, w_f, dw):
        return (w_in - dw / 2  + nu * (w_f - w_in + dw)) #* (self.n) / (self.n-1)   #shift
    
    def get_spectrum(self, left_lim, right_lim, lam, num_nu = 250, threshold = 1e-2, w_exp = [], plot = True, return_coeff = False):
        dw, x = self.scale_signal(self.z - 1 / 2, self.t[-1], left_lim , right_lim)
        p = self.get_polynomial(x, noise=True, lam=lam)
        nu = np.linspace(0, 1, num_nu)
        P = np.abs([np.sum(np.exp(-1j * 2 * np.pi * v * np.arange(np.size(x))) * p) for v in nu])
        P /= P.max()
        omega_Anm = self.scale_freq(nu, left_lim, right_lim, dw)

        filter = np.abs(1 - P)
        omega_ordered =  omega_Anm[np.argsort(filter)]
        P_ordered = P[np.argsort(filter)]
        omega_k = omega_ordered[np.abs(1 - P_ordered) < threshold]
        P_k = P_ordered[np.abs(1 - P_ordered) < threshold]

        if plot:
            plt.xlabel(r"$\omega$")
            plt.ylabel(r"$|P(\omega)|$")
            plt.plot(omega_Anm, P)
            plt.axhline(y=1, color = "green")
            if w_exp:
                for w in w_exp:
                    plt.axvline(x=w, color="red", ls=":")
                    plt.axvline(x=-w, color="red", ls=":")

        if return_coeff:
            return omega_k, P_k
        return omega_k

# Analysis Error
def remove_near_freq(omega_k, P_k, threshold=0.01 * 2 * np.pi):

    idx_sorted = np.argsort(omega_k)
    omega_sorted = omega_k[idx_sorted]
    P_sorted = P_k[idx_sorted]

    keep = np.ones(len(omega_sorted), dtype=bool)

    for i in range(len(omega_sorted)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(omega_sorted)):
            if not keep[j]:
                continue
            if np.abs(omega_sorted[j] - omega_sorted[i]) < threshold:
                if P_sorted[i] >= P_sorted[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break  

    return omega_sorted[keep]

def fit_cosine_sum(t, signal, omega):
    A = np.column_stack([np.cos(w * t) for w in omega])
    c_fit, residuals, rank, s = np.linalg.lstsq(A, signal, rcond=None)
    return c_fit



def get_ck(time, signal, omega_k, max_ck=1):
    signal = 2 * signal - 1    
    A = np.column_stack([np.cos(w * time) for w in omega_k])
    bounds = (0, max_ck)
    result = lsq_linear(A, signal, bounds=bounds)
    return result.x

#def get_ck(time, signal, omega_k, max_ck = 1):                 ######## for Gaussian
#     signal = 2 * signal - 1    
#     A = np.column_stack([np.cos(w * time) for w in omega_k])
#     c_fit, residuals, rank, s = np.linalg.lstsq(A, signal, rcond=None)
#     c_fit[c_fit > max_ck] = 1
#     return c_fit



def get_error(c_k, omega_k, c_k_exp, omega_k_exp):
    match_omega_k = []
    match_c_k = []
    for omega_expected in omega_k_exp:
         index = np.argmin(np.abs(omega_k - omega_expected))
         match_omega_k.append(omega_k[index])
         match_c_k.append(c_k[index])
    
    match_omega_k = np.array(match_omega_k)
    match_c_k = np.array(match_c_k)
    error_omega = np.sum(np.abs(match_c_k) * np.abs(match_omega_k - omega_k_exp)) / np.sum(np.abs(c_k_exp * omega_k_exp))
    error_c_k = np.sum(np.abs(c_k_exp - match_c_k)) / np.sum(np.abs(c_k_exp))
    error_noise = 0


    index_noise = ~np.isin(omega_k, match_omega_k)
    omega_k_noise = omega_k[index_noise]
    c_k_noise = c_k[index_noise]
    if omega_k_noise.tolist() != []:
        error_noise = np.sum(np.abs(c_k_noise * omega_k_noise)) / np.sum(c_k_exp * omega_k_exp)

    return error_omega + error_c_k + error_noise


def study_error(time, signal, c_k_exp, omega_k_exp, lam, left_lim = -np.pi, right_lim = np.pi, dt = 1, num_vu = 250) :
    num_points = np.size(signal) * dt
    Anm = Atomic_Norm_Minimization(time, signal)
    dw, x = Anm.scale_signal(signal - 1 / 2, num_points / 2, left_lim , right_lim )
    p = Anm.get_polynomial(x, noise=True, lam=lam)
    nu = np.linspace(0, 1, num_vu)
    P = np.abs([np.sum(np.exp(-1j * 2 * np.pi * v * np.arange(np.size(x))) * p) for v in nu])
    P /= P.max()
    omega_Anm = Anm.scale_freq(nu, left_lim, right_lim, dw)

    filter = np.abs(1 - P)
    omega_ordered =  omega_Anm[np.argsort(filter)]
    P_ordered = P[np.argsort(filter)]
    omega_k = omega_ordered[np.abs(1 - P_ordered) < 0.05]
    P_k = P_ordered[np.abs(1 - P_ordered) < 0.05]

    omega_k = remove_near_freq(omega_k, P_k)
    omega_k = omega_k[omega_k > 0]
    c_k = get_ck(time, signal, omega_k)        #######
    return get_error(c_k, omega_k, c_k_exp, omega_k_exp), omega_k, c_k