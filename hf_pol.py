import arc
import numpy as np
from matplotlib.figure import Figure
import math
from sympy import S
from sympy.physics.wigner import wigner_6j
import math

class HFPolarizabilityCalculator:
    def __init__(self, atom_name, n, L, J, F, mF, q):
        self.atom_name = atom_name
        self.atom = self._return_atom()
        self.n = n
        self.L = L
        self.J = J
        self.F = F
        self.mF = mF
        self.q = q
        self.pol = arc.DynamicPolarizability(self.atom, self.n, self.L, self.J)
        self.pol.defineBasis(5, 15)
    
    def _return_atom(self):
        if self.atom_name == "Rb85":
            return arc.Rubidium85()
        elif self.atom_name == "Rb87":
            return arc.Rubidium87()
        else:
            print(f"Unknown atom choice: {self.atom_name}. Defaulting to Rb85.")
            return arc.Rubidium85()

    def _minus_one_pow(self, x):
        """
        Returns (-1)^x for integer x, with mild float protection.
        """
        n = int(round(float(x)))
        return -1.0 if (n % 2) else 1.0


    def _arc_to_irreducible(self, J, a_scalar, a_vector, a_tensor):
        """
        Convert ARC's fine-structure scalar/vector/tensor polarizabilities
        into irreducible rank-K components alpha^(K).

        ARC returns conventional scalar/vector/tensor components.
        """
        # rank-0
        alpha0_irred = math.sqrt(3.0 * (2.0 * J + 1.0)) * a_scalar

        # rank-1
        if J <= 0:
            alpha1_irred = 0.0
        else:
            alpha1_irred = -math.sqrt((J + 1.0) * (2.0 * J + 1.0) / (2.0 * J)) * a_vector

        # rank-2
        # For J = 1/2 this vanishes in the fine-structure-only treatment.
        if J < 1.0:
            alpha2_irred = 0.0
        else:
            alpha2_irred = -math.sqrt(
                3.0 * (J + 1.0) * (2.0 * J + 1.0) * (2.0 * J + 3.0)
                / (2.0 * J * (2.0 * J - 1.0))
            ) * a_tensor

        return alpha0_irred, alpha1_irred, alpha2_irred


    def hyperfine_polarizability_from_arc(self, J, F, mF, q,
                                        a_scalar, a_vector, a_tensor, a_core,
                                        I=2.5):
        """
        Effective polarizability for a hyperfine Zeeman state |F, mF>
        for pure spherical polarization q = -1, 0, +1.

        Parameters
        ----------
        J : float
            Electronic angular momentum of the fine-structure manifold.
        F : float
            Hyperfine total angular momentum.
        mF : float
            Projection of F.
        q : int
            Polarization component: -1 (sigma-), 0 (pi), +1 (sigma+).
        a_scalar, a_vector, a_tensor, a_core : float
            Outputs from ARC getPolarizability(...):
                ret[0], ret[1], ret[2], ret[3]
        I : float
            Nuclear spin. For 85Rb, I = 5/2 = 2.5.

        Returns
        -------
        dict with:
            alpha_F_scalar
            alpha_F_vector
            alpha_F_tensor
            alpha_total
        """
        if q not in (-1, 0, 1):
            raise ValueError("q must be -1, 0, or +1")

        if abs(mF) > F:
            raise ValueError("|mF| must be <= F")

        # Allowed hyperfine values for given I,J
        Fmin = abs(I - J)
        Fmax = I + J
        if F < Fmin - 1e-12 or F > Fmax + 1e-12:
            raise ValueError(f"F must lie in [{Fmin}, {Fmax}] for I={I}, J={J}")

        # Total scalar fine-structure polarizability includes the core
        alphaJ_scalar_total = a_scalar + a_core

        # Convert ARC vector/tensor conventions to irreducible components
        # Keep scalar separately since alpha_F^S = alpha_J^S in this approximation.
        _, alpha1_irred, alpha2_irred = self._arc_to_irreducible(J, a_scalar, a_vector, a_tensor)

        phase = self._minus_one_pow(J + I + F)

        # Hyperfine scalar piece
        alphaF_scalar = alphaJ_scalar_total

        # Hyperfine vector piece
        if F == 0:
            alphaF_vector = 0.0
        else:
            six1 = float(wigner_6j(S(F), 1, S(F), S(J), S(I), S(J)).evalf())
            alphaF_vector = (
                phase
                * math.sqrt(2.0 * F * (2.0 * F + 1.0) / (F + 1.0))
                * six1
                * alpha1_irred
            )

        # Hyperfine tensor piece
        if F < 1.0 or J < 1.0:
            alphaF_tensor = 0.0
        else:
            six2 = float(wigner_6j(S(F), 2, S(F), S(J), S(I), S(J)).evalf())
            alphaF_tensor = (
                -phase
                * math.sqrt(
                    2.0 * F * (2.0 * F - 1.0) * (2.0 * F + 1.0)
                    / (3.0 * (F + 1.0) * (2.0 * F + 3.0))
                )
                * six2
                * alpha2_irred
            )

        # For pure q:
        # q = -1 -> sigma-
        # q =  0 -> pi
        # q = +1 -> sigma+
        #
        # Using C = |u_-1|^2 - |u_+1|^2 = -q
        # and D = 1 - 3|u_0|^2 = 1 for sigma±, -2 for pi
        C = -float(q)
        D = 1.0 - 3.0 * (1.0 if q == 0 else 0.0)

        alpha_total = alphaF_scalar

        if F != 0:
            alpha_total += C * (mF / (2.0 * F)) * alphaF_vector

        if F >= 1.0:
            alpha_total -= D * (
                (3.0 * mF * mF - F * (F + 1.0))
                / (2.0 * F * (2.0 * F - 1.0))
            ) * alphaF_tensor

        return {
            "alpha_F_scalar": alphaF_scalar,
            "alpha_F_vector": alphaF_vector,
            "alpha_F_tensor": alphaF_tensor,
            "alpha_total": alpha_total,
        }
    
    def calculate(self, wavelength):
        self.arc_results = self.pol.getPolarizability(wavelength, units='SI')
        a_scalar = float(self.arc_results[0])
        a_vector = float(self.arc_results[1])
        a_tensor = float(self.arc_results[2])
        a_core = float(self.arc_results[3])
        self.alphas = self.hyperfine_polarizability_from_arc(
            self.J, self.F, self.mF, self.q,
            a_scalar, a_vector, a_tensor, a_core
        )
        return self.alphas['alpha_total']