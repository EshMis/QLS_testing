"""Selected-pipeline operation-count proxies with explicit provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from qls_testing.core.datatypes import IntegrationResult, LinearizedSystem
from qls_testing.core.interfaces import ComplexityEstimator


@dataclass(frozen=True)
class ComplexityTerm:
    """One symbolic cost term evaluated with values from the current run."""

    stage: str
    label: str
    symbolic: str
    evaluated: str
    source: str
    assumptions: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ComplexityReport:
    metrics: dict[str, float | int | str]
    asymptotic: dict[str, str]
    terms: tuple[ComplexityTerm, ...] = ()
    caveat: str = (
        "Operation-count proxies are not wall-clock or quantum-advantage claims; backend, "
        "compilation, precision, state preparation, and readout can dominate."
    )

    @property
    def stages(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(term.stage for term in self.terms))

    def to_dict(self) -> dict[str, object]:
        return {
            "metrics": self.metrics,
            "asymptotic": self.asymptotic,
            "terms": [term.to_dict() for term in self.terms],
            "stages": self.stages,
            "caveat": self.caveat,
        }


class DefaultComplexityEstimator(ComplexityEstimator):
    """Build a provenance-preserving report for only the selected stages."""

    def estimate(
        self,
        *,
        lifted: LinearizedSystem,
        integration: IntegrationResult,
        integrator_name: str,
        solver_name: str,
    ) -> ComplexityReport:
        dimension = int(lifted.matrix.shape[0])
        nnz = int(np.count_nonzero(lifted.matrix))
        solves = len(integration.solve_diagnostics)
        steps = max(len(integration.times) - 1, 0)
        physical_dimension = int(lifted.physical_dimension)
        order = int(lifted.metadata.get("order", 1))
        terms: list[ComplexityTerm] = []
        asymptotic: dict[str, str] = {}

        terms.append(
            ComplexityTerm(
                "Linearization",
                "Lifted dimension",
                r"D(n,N)=\sum_{k=1}^{N}{n+k-1\choose k}+\delta_{\rm affine}",
                f"n={physical_dimension}, N={order} -> D={dimension}",
                "Carleman monomial enumeration in linearization/carleman.py.",
                "Commutative monomials through degree N; one affine coordinate only when needed.",
            )
        )
        terms.append(
            ComplexityTerm(
                "Linearization",
                "Lifted operator construction/storage",
                r"C_{\rm lift}=O(D^2)\ {\rm dense},\qquad O(\operatorname{nnz}(A_C))\ {\rm sparse}",
                f"D={dimension}, nnz(A_C)={nnz}, dense slots={dimension**2}",
                "Each retained derivative monomial contributes an entry to the Carleman operator.",
                "The implementation constructs a dense array but reports its realized nonzero count.",
            )
        )
        asymptotic["linearization"] = "O(D^2) dense construction/storage; O(nnz(A_C)) sparse storage"
        if integration.metadata.get("method") == "adaptive_restarted_carleman":
            segment_orders = tuple(integration.metadata.get("orders", ()))
            terms.append(
                ComplexityTerm(
                    "Linearization",
                    "Adaptive restart/order trials",
                    r"C_{\rm adapt}=\sum_{j=1}^{N_{\rm seg}}[C(N_j)+C(N_j+1)]",
                    f"segments={len(segment_orders)}, accepted orders={segment_orders}",
                    "Each segment compares adjacent Carleman orders and re-embeds its accepted endpoint.",
                    "The highest-order cap may terminate a segment without an N+1 comparison.",
                )
            )

        if integrator_name in {"backward_euler", "crank_nicolson", "bdf2"}:
            terms.append(
                ComplexityTerm(
                    "Integrator",
                    f"{integrator_name} step loop",
                    r"C_{\rm int}=N_{\rm solve}\,[C_{\rm rhs}+C_{\rm solve}(D)]",
                    f"N_step={steps}, N_solve={solves}, D={dimension}, rhs matvec <= {steps * nnz}",
                    "One linear-system call per recorded fixed step (BDF2 includes its bootstrap solve).",
                    "A time-independent left-hand matrix can be factorized/reused by a capable backend.",
                )
            )
            asymptotic["integrator"] = "O(N_step*nnz(A_C)) rhs work plus N_solve selected-solver calls"
        elif integrator_name in {
            "folded_backward_euler", "folded_crank_nicolson", "folded_bdf2"
        }:
            folded_dimension = int(integration.metadata.get("folded_dimension", steps * dimension))
            folded_nnz = int(integration.metadata.get("folded_nnz", 0))
            kron_terms = int(integration.metadata.get("kronecker_term_count", 0))
            terms.append(
                ComplexityTerm(
                    "Integrator",
                    f"{integrator_name} all-at-once system",
                    r"D_F=N_{\rm step}D,\qquad M_F=\sum_{j=1}^{r}T_j\otimes S_j",
                    f"D_F={folded_dimension}, nnz={folded_nnz}, Kronecker terms={kron_terms}; one QLS call",
                    "Sparse time-shift/state-operator assembly in hardware_path/folded_systems.py.",
                    "The current solver plugin boundary becomes dense; hardware work should compile the Kronecker terms directly.",
                )
            )
            asymptotic["integrator"] = "one QLS solve at folded dimension N_step*D"
        elif integrator_name == "pade22":
            terms.append(
                ComplexityTerm(
                    "Integrator",
                    "Global Padé [2/2] evaluations",
                    r"C_{\rm int}=C(A_C^2)+N_t[C_{\rm matvec}+C_{\rm solve}(D)]",
                    f"N_t={steps}, D={dimension}, solves={solves}",
                    "A_C^2 is formed once; each output time builds numerator/denominator and solves.",
                    "Dense A_C^2 costs O(D^3); sparse fill-in can change that estimate.",
                )
            )
            asymptotic["integrator"] = "O(D^3) setup plus N_t selected-solver calls"
        elif integrator_name == "rk45":
            nfev = int(integration.metadata.get("function_evaluations", 0))
            terms.append(
                ComplexityTerm(
                    "Integrator",
                    "Adaptive RK45 matrix-vector products",
                    r"C_{\rm RK45}=N_{\rm fev}\,O(\operatorname{nnz}(A_C))",
                    f"N_fev={nfev}, nnz(A_C)={nnz}, proxy={nfev * nnz}",
                    "Every lifted RHS evaluation computes A_C y; SciPy reports N_fev.",
                    "Sparse matvec model; dense execution is O(N_fev D^2). No QLS is called.",
                )
            )
            asymptotic["integrator"] = "O(N_fev*nnz(A_C)) sparse or O(N_fev*D^2) dense"
        elif integrator_name == "krylov_exponential":
            terms.append(
                ComplexityTerm(
                    "Integrator",
                    "Krylov exponential action",
                    r"C_{\rm Krylov}\approx O(N_t\,m\,\operatorname{nnz}(A_C))",
                    f"N_t={len(integration.times)}, nnz(A_C)={nnz}; internal Krylov degree m is SciPy-selected",
                    "expm_multiply builds a Krylov/Taylor approximation using repeated sparse matvecs.",
                    "m depends on tolerance, norm, spectral structure, and internal scaling.",
                )
            )
            asymptotic["integrator"] = "approximately O(N_t*m*nnz(A_C))"
        elif integrator_name == "exponential":
            terms.append(
                ComplexityTerm(
                    "Integrator",
                    "Dense matrix-exponential reference",
                    r"C_{\rm exp}=N_t\,O(D^3)",
                    f"N_t={len(integration.times)}, D={dimension}",
                    "The reference path evaluates scipy.linalg.expm(t A_C) at every output time.",
                    "Dense scaling-and-squaring proxy; this path intentionally does not call a QLS.",
                )
            )
            asymptotic["integrator"] = "N_t dense matrix exponentials, approximately O(N_t*D^3)"
        elif integrator_name in {"lindblad_ndme", "pennylane_lindblad"}:
            density_dimension = int(integration.metadata.get("density_dimension", 2 * dimension))
            liouville_dimension = int(integration.metadata.get("liouville_dimension", density_dimension**2))
            terms.append(
                ComplexityTerm(
                    "Lindbladian / NDME",
                    "Density-matrix embedding",
                    r"d_\rho=2D,\qquad d_{\mathcal L}=d_\rho^2",
                    f"D={dimension}, d_rho={density_dimension}, d_L={liouville_dimension}",
                    "NDME uses a two-block ancilla extension and vectorizes the density matrix.",
                    "This branch evolves a density matrix and does not invoke a QLS solver.",
                )
            )
            if integrator_name == "pennylane_lindblad":
                applications = int(integration.metadata.get("channel_applications", 0))
                qubits = int(integration.metadata.get("qubits", 0))
                terms.append(
                    ComplexityTerm(
                        "Lindbladian / NDME",
                        "PennyLane short-time channel execution",
                        r"C_{\rm channel}=N_{\rm interval}N_{\rm substep}C_{\rm Kraus}(n_q)",
                        f"channel applications={applications}, qubits={qubits}",
                        "Each substep applies a normalized first-order Kraus channel on default.mixed.",
                        "Mixed-state simulation stores O(4^n_q) amplitudes; hardware channel realization needs dilation or native noise.",
                    )
                )
            else:
                terms.append(
                    ComplexityTerm(
                        "Lindbladian / NDME",
                        "Classical Liouvillian integration proxy",
                        r"C_{\rm NDME,classical}\sim N_{\rm fev}\,O(d_{\mathcal L}^{2})",
                        f"Liouville dimension={liouville_dimension}; dense slots={liouville_dimension**2}",
                        "Vectorized master-equation RHS applies the Liouvillian to vec(rho).",
                        "This is the repository's classical simulation cost, not the paper's oracle complexity.",
                    )
                )
            asymptotic["lindblad_ndme"] = (
                "short-time PennyLane Kraus channels" if integrator_name == "pennylane_lindblad"
                else "dense classical propagation on Liouville dimension (2D)^2"
            )

        diagnostics = integration.solve_diagnostics
        metadata = [item.metadata for item in diagnostics]
        conditions = [
            float(item["condition_number"])
            for item in metadata
            if "condition_number" in item
        ]
        condition = max(conditions, default=float(integration.metadata.get("lhs_condition", 1.0)))
        if solves and solver_name != "not_applicable":
            terms.extend(self._solver_terms(solver_name, metadata, solves, dimension, condition))
            asymptotic["solver"] = terms[-1].symbolic

        metrics: dict[str, float | int | str] = {
            "lifted_dimension": dimension,
            "nnz": nnz,
            "sparsity": float(1.0 - nnz / max(dimension * dimension, 1)),
            "recorded_steps": steps,
            "linear_solves": solves,
            "condition_number_proxy": condition,
            "integrator": integrator_name,
            "solver": solver_name,
            "selected_stage_count": len(tuple(dict.fromkeys(term.stage for term in terms))),
        }
        return ComplexityReport(metrics, asymptotic, tuple(terms))

    @staticmethod
    def _solver_terms(
        solver_name: str,
        metadata: list[dict[str, object]],
        solves: int,
        dimension: int,
        condition: float,
    ) -> list[ComplexityTerm]:
        if solver_name == "classical":
            return [ComplexityTerm(
                "Linear solver", "Dense direct solve",
                r"C_{\rm solve}=O(D^3)+N_{\rm rhs}O(D^2)",
                f"D={dimension}, N_rhs={solves}; no factorization reuse is implemented",
                "NumPy solve performs a dense factorization for each solver call.",
                "A cached factorization would reduce repeated right-hand sides to O(D^2) each.",
            )]
        if solver_name.startswith("pennylane") and "vqls" in solver_name:
            iterations = sum(int(item.get("steps", 0)) for item in metadata)
            depths = [int(item.get("circuit_depth", 0)) for item in metadata]
            shots = max((int(item.get("shots", 0)) for item in metadata), default=0)
            return [ComplexityTerm(
                "Linear solver", "PennyLane VQLS optimization and readout",
                r"C_{\rm VQLS}\sim N_{\rm opt}\,N_{\rm QNode/step}\,C_{\rm circuit}+N_{\rm solve}N_{\rm shots}",
                f"total optimizer steps={iterations}, QNode traversals/step~=3, max depth={max(depths, default=0)}, solves={solves}, shots/readout={shots}",
                "Each step records one explicit gradient, Adam computes its update gradient, and the loss is reevaluated; shot QNodes are used for final probability diagnostics.",
                "Statevector simulation cost is exponential in qubits; hardware cost also includes state preparation and measurement precision.",
            )]
        if solver_name == "preconditioned_qsvt":
            nested = [item.get("base_metadata", {}) for item in metadata]
            degrees = [int(item.get("degree", 0)) for item in nested if isinstance(item, dict)]
            degree = max(degrees, default=0)
            return [
                ComplexityTerm(
                    "Preconditioner", "Jacobi left scaling",
                    r"C_{\rm Jacobi}=O(D)+O(\operatorname{nnz}(M))",
                    f"D={dimension}, solves={solves}, condition proxy after scaling={condition:.3g}",
                    "Each solve forms diag(M)^(-1)M and diag(M)^(-1)b before QSVT.",
                    "Requires nonzero diagonal; access/preparation cost is not free on hardware.",
                ),
                ComplexityTerm(
                    "Linear solver", "Preconditioned QSVT inverse polynomial",
                    r"Q=N_{\rm solve}d_p",
                    f"solves={solves}, observed degree={degree}, query proxy={solves * degree}",
                    "The QSVT simulator is applied to the Jacobi-scaled operator.",
                    "Degree is an inverse-polynomial proxy; simulator SVD work is classical.",
                ),
            ]
        if "qsvt" in solver_name:
            degrees = [int(item.get("degree", 0)) for item in metadata]
            degree = max(degrees, default=0)
            return [ComplexityTerm(
                "Linear solver", "QSVT inverse polynomial",
                r"Q_{\rm QSVT}=N_{\rm solve}\,O(\kappa\log(1/\epsilon))\approx N_{\rm solve}d_p",
                f"solves={solves}, observed degree={degree}, query proxy={solves * degree}, kappa proxy={condition:.3g}",
                "A degree-d_p singular-value polynomial alternates block-encoding and phase operations.",
                "Ideal query model assumes efficient block encoding, state preparation, and bounded singular values.",
            )]
        if "hhl" in solver_name:
            clocks = [int(item.get("n_clock", 0)) for item in metadata]
            clock = max(clocks, default=0)
            return [ComplexityTerm(
                "Linear solver", "HHL phase estimation",
                r"Q_{\rm HHL}\approx N_{\rm solve}(2^{n_c}-1)\ \text{controlled evolutions}",
                f"solves={solves}, clock qubits={clock}, controlled-evolution proxy={solves * (2**clock - 1 if clock else 0)}",
                "n_c phase-estimation qubits apply controlled U^(2^j) powers.",
                "Ideal HHL scaling additionally depends on sparsity, condition number, precision, preparation, and postselection/amplitude amplification.",
            )]
        if solver_name == "iterative_refinement":
            rounds = sum(int(item.get("iterations", 0)) for item in metadata)
            base_methods = {str(item.get("base_method", "configured base solver")) for item in metadata}
            return [
                ComplexityTerm(
                    "Iterative refinement", "Residual correction loop",
                    r"C_{\rm IR}=C_{\rm base}+N_{\rm corr}(C_{\rm residual}+C_{\rm base})",
                    f"outer solves={solves}, correction rounds={rounds}, base={sorted(base_methods)}",
                    "Each round forms b-Mx and asks the configured base solver for a correction.",
                    "Convergence requires each approximate correction to reduce the true residual.",
                ),
                ComplexityTerm(
                    "Linear solver", "Configured refinement base solver calls",
                    r"N_{\rm base}=N_{\rm solve}+N_{\rm corr}",
                    f"base calls={solves + rounds}; methods={sorted(base_methods)}",
                    "One initial base solve plus one base solve per recorded correction round.",
                    "Backend-specific degree/query details are retained in correction metadata when available.",
                ),
            ]
        if solver_name == "vqls_simulator":
            iterations = sum(int(item.get("iterations", 0)) for item in metadata)
            return [ComplexityTerm(
                "Linear solver", "Classical full-vector VQLS surrogate",
                r"C\sim N_{\rm iter}[O(D^2)+C_{\rm BFGS}(D)]",
                f"solves={solves}, total BFGS iterations={iterations}, D={dimension}",
                "Each objective reconstructs a normalized dense vector and evaluates Mx-b.",
                "This is a classical optimizer surrogate, not a gate/query estimate.",
            )]
        return [ComplexityTerm(
            "Linear solver", solver_name,
            r"C_{\rm solver}=N_{\rm solve}\,C_{\rm selected\ backend}(D)",
            f"N_solve={solves}, D={dimension}, condition proxy={condition:.3g}",
            "Solver-call count comes directly from integration diagnostics.",
            "See the selected solver metadata and method documentation for backend-specific assumptions.",
        )]
