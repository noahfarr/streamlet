"""Closed-form moments and stability quantities for the rare-spike linear TD(0) demo.

Feature phi in {M w.p. p, 1 w.p. 1-p}, reward r == 0, lambda == 0, so the trace is
z = phi and the per-sample interaction is X = phi (phi - gamma phi'), with phi and phi'
independent draws (i.i.d. states). All quantities are pure functions of (M, p, gamma).
"""

from dataclasses import dataclass


def feature_moment(k: float, M: float, p: float) -> float:
    """E[phi^k] = p M^k + (1 - p)."""
    return p * (M**k) + (1.0 - p)


def mean_interaction(M: float, p: float, gamma: float) -> float:
    """A = E[X] = E[phi^2] - gamma E[phi]^2 (phi independent of phi')."""
    e1 = feature_moment(1, M, p)
    e2 = feature_moment(2, M, p)
    return e2 - gamma * e1 * e1


def second_moment_interaction(M: float, p: float, gamma: float) -> float:
    """E[X^2] = E[phi^4] - 2 gamma E[phi^3] E[phi] + gamma^2 E[phi^2]^2."""
    e1 = feature_moment(1, M, p)
    e2 = feature_moment(2, M, p)
    e3 = feature_moment(3, M, p)
    e4 = feature_moment(4, M, p)
    return e4 - 2.0 * gamma * e3 * e1 + (gamma**2) * e2 * e2


def cv2(M: float, p: float, gamma: float) -> float:
    """Squared coefficient of variation of X: Var(X)/E[X]^2 = E[X^2]/A^2 - 1."""
    a = mean_interaction(M, p, gamma)
    ex2 = second_moment_interaction(M, p, gamma)
    return ex2 / (a * a) - 1.0


def rho(alpha: float, M: float, p: float, gamma: float) -> float:
    """Per-step second-moment multiplier rho(alpha) = 1 - 2 alpha A + alpha^2 E[X^2]."""
    a = mean_interaction(M, p, gamma)
    ex2 = second_moment_interaction(M, p, gamma)
    return 1.0 - 2.0 * alpha * a + alpha * alpha * ex2


def alpha_mean_step(M: float, p: float, gamma: float) -> float:
    """Mean-stability-centred step alpha = 1 / A.

    This is the idealized expectation-based step the construction labels "AlphaBound":
    it steps at the inverse of the *mean* interaction, giving rho = CV^2. The literal
    Dabney & Barto AlphaBound instead caps at the inverse of each *instantaneous*
    interaction (see streax.optimizers.AlphaBound), which is why the literal method does
    not diverge here.
    """
    return 1.0 / mean_interaction(M, p, gamma)


def alpha_calibrated(M: float, p: float, gamma: float) -> float:
    """Variance-optimal step alpha* = A / E[X^2] (Calibrated with nu = 0)."""
    return mean_interaction(M, p, gamma) / second_moment_interaction(M, p, gamma)


def mean_threshold(M: float, p: float, gamma: float) -> float:
    """Largest mean-stable step: alpha < 2 / A."""
    return 2.0 / mean_interaction(M, p, gamma)


def ms_threshold(M: float, p: float, gamma: float) -> float:
    """Largest mean-square-stable step: alpha < 2 A / E[X^2]."""
    a = mean_interaction(M, p, gamma)
    ex2 = second_moment_interaction(M, p, gamma)
    return 2.0 * a / ex2


@dataclass(frozen=True)
class Stability:
    M: float
    p: float
    gamma: float
    A: float
    E_X2: float
    cv2: float
    alpha_mean_step: float
    alpha_calibrated: float
    rho_mean_step: float
    rho_calibrated: float
    mean_threshold: float
    ms_threshold: float


def stability(M: float, p: float, gamma: float) -> Stability:
    a = mean_interaction(M, p, gamma)
    ex2 = second_moment_interaction(M, p, gamma)
    c = ex2 / (a * a) - 1.0
    amean = alpha_mean_step(M, p, gamma)
    acal = alpha_calibrated(M, p, gamma)
    return Stability(
        M=M,
        p=p,
        gamma=gamma,
        A=a,
        E_X2=ex2,
        cv2=c,
        alpha_mean_step=amean,
        alpha_calibrated=acal,
        rho_mean_step=rho(amean, M, p, gamma),
        rho_calibrated=rho(acal, M, p, gamma),
        mean_threshold=mean_threshold(M, p, gamma),
        ms_threshold=ms_threshold(M, p, gamma),
    )


if __name__ == "__main__":
    s = stability(M=3.0, p=0.03, gamma=0.0)
    print(f"M={s.M} p={s.p} gamma={s.gamma}")
    print(f"  A           = {s.A:.4f}")
    print(f"  E[X^2]      = {s.E_X2:.4f}")
    print(f"  E[X^2]/A^2  = {s.E_X2 / s.A**2:.4f}  (= 1 + CV^2)")
    print(f"  CV^2        = {s.cv2:.4f}")
    print(f"  alpha_mean  = {s.alpha_mean_step:.4f}   rho_mean  = {s.rho_mean_step:.4f}")
    print(f"  alpha_Cal   = {s.alpha_calibrated:.4f}   rho_Cal   = {s.rho_calibrated:.4f}")
