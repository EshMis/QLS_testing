"""LaTeX snippets describing only the components active in a configuration."""

from __future__ import annotations

from qls_testing.core.config import Config


def active_math_sections(config: Config) -> tuple[tuple[str, str, str], ...]:
    sections: list[tuple[str, str, str]] = []
    if config.system.name == "lindblad_enzyme_ndme":
        sections.extend(
            (
                (
                    "Linear ODE convention",
                    r"\dot{\mu}=A\mu=-V\mu,\qquad V=-A",
                    "The reduced enzyme Carleman operator is embedded without using a QLS.",
                ),
                (
                    "NDME Lindbladian",
                    r"B=\frac{V+V^\dagger}{2},\quad H_1=\frac{V-V^\dagger}{2i},\quad G^\dagger G=2(B+\kappa I)",
                    "A minimal shift κ makes the Hermitian part positive semidefinite.",
                ),
                (
                    "Off-diagonal extraction",
                    r"\rho_{01}(t)=\frac12 e^{(A-\kappa I)t}|\mu_0\rangle\langle\mu_0|,\quad \mu(t)=2\|\mu_0\|e^{\kappa t}\rho_{01}(t)|\mu_0\rangle",
                    "This is the nondiagonal density-matrix encoding of the supplied PDF.",
                ),
                (
                    "NDME square-root-access complexity",
                    r"\widetilde O\!\left(\eta_T^{-1}\alpha_VT\frac{\log(1/\epsilon)}{\log\log(1/\epsilon)}\right)",
                    "Time-independent oracle complexity from Theorem 2 of the supplied reference.",
                ),
            )
        )
    else:
        sections.append(
            (
                "Carleman embedding",
                r"y_\alpha=x^\alpha,\qquad \dot y_\alpha=\sum_i \alpha_i x^{\alpha-e_i}f_i(x),\qquad 1\le |\alpha|\le N",
                "Terms above the configured order N are truncated.",
            )
        )
        integrator_math = {
            "backward_euler": r"(I-\Delta t A)y_{k+1}=y_k",
            "folded_backward_euler": r"\mathcal M\,(y_1,\ldots,y_K)^T=(y_0,0,\ldots,0)^T",
            "crank_nicolson": r"(I-\tfrac{\Delta t}{2}A)y_{k+1}=(I+\tfrac{\Delta t}{2}A)y_k",
            "bdf2": r"(\tfrac32 I-\Delta t A)y_{k+1}=2y_k-\tfrac12y_{k-1}",
            "pade22": r"(I-\tfrac t2A+\tfrac{t^2}{12}A^2)y(t)=(I+\tfrac t2A+\tfrac{t^2}{12}A^2)y_0",
            "rk45": r"\|e_{\mathrm{embedded}}\|\le \mathrm{atol}+\mathrm{rtol}\,\|y\|",
            "krylov_exponential": r"y(t)\approx \|y_0\|V_m e^{tH_m}e_1,\qquad H_m=V_m^\dagger A V_m",
            "exponential": r"y(t)=e^{tA}y_0",
        }
        sections.append(
            (
                f"Integrator: {config.integrator.name}",
                integrator_math.get(config.integrator.name, r"\dot y=Ay"),
                "This is the actual selected time-evolution rule.",
            )
        )
        solver_math = {
            "classical": r"Mx=b,\qquad x=M^{-1}b",
            "hhl_simulator": r"|x\rangle\propto\sum_j \frac{\beta_j}{\lambda_j}|u_j\rangle",
            "pennylane_hhl": r"\mathrm{QPE}(e^{2\pi iM/s})\rightarrow R_y\!\left(2\arcsin\frac{C}{\lambda}\right)\rightarrow\mathrm{QPE}^{\dagger}",
            "qsvt_simulator": r"p(\sigma)\approx\sigma^{-1},\qquad x\approx Vp(\Sigma)U^\dagger b",
            "pennylane_qsvt": r"(\langle0|\otimes I)U_{\mathrm{QSVT}}(|0\rangle\otimes I)\approx C M^{-1}",
            "pennylane_vqls": r"\min_\theta\frac{\|M\,s(\theta)u(\theta)-b\|^2}{\|b\|^2},\quad s=\frac{\langle Mu,b\rangle}{\langle Mu,Mu\rangle}",
            "pennylane_complex_vqls": r"\min_\theta\frac{\|M\,s(\theta)u(\theta)-b\|^2}{\|b\|^2},\quad u(\theta)\in\mathbb C^D",
            "iterative_refinement": r"r_k=b-Mx_k,\quad M\delta_k\approx r_k,\quad x_{k+1}=x_k+\delta_k",
        }
        sections.append(
            (
                f"Linear solver: {config.qls.name}",
                solver_math.get(config.qls.name, r"Mx\approx b"),
                "Iterative refinement belongs to the linear solver, not the integrator.",
            )
        )
    sections.extend(
        (
            (
                "Lift dimension",
                r"D(n,N)=\sum_{k=1}^{N}{n+k-1\choose k}=O(n^N)\quad(N\ \mathrm{fixed})",
                "An affine system adds one homogeneous coordinate.",
            ),
            (
                "Dense and sparse evolution",
                r"C_{\mathrm{dense}}=O(N_{\mathrm{steps}}D^3),\qquad C_{\mathrm{Krylov}}\approx O(m\,\mathrm{nnz}(A))",
                "Factorization reuse can reduce repeated dense solves to O(D³+Nsteps D²).",
            ),
            (
                "Idealized quantum dependence",
                r"C_{\mathrm{HHL}}=\widetilde O\!\left(\frac{s\kappa\,\mathrm{polylog}(D)}{\epsilon}\right),\qquad C_{\mathrm{QSVT}}=O\!\left(\kappa\log\frac1\epsilon\right)\ \mathrm{queries}",
                "State preparation, postselection, readout, noise, and compilation are not free.",
            ),
        )
    )
    return tuple(sections)
