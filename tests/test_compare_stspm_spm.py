import numpy as np
from stspm import sec2yr

class RiverSPMTest:

    def __init__(self, riverstspm, a=2/3, ke_PA=1e-4):
        self.stspm = riverstspm
        self.a, self.ke_PA = a, ke_PA
        Rmean_x = self.stspm.calculate_average_runoff()
        PA = 1e3 * sec2yr(1)**2
        self.ke = ke_PA / PA**a                       # m/yr per (g m^-1 yr^-2)^a
        s = self.stspm
        self.n = s.beta * a
        self.m = s.alpha * (1 - s.wa) * a
        self.K_x = self.ke * s.kw**(-s.alpha*a) * s.kt**a * Rmean_x**(s.alpha*(1-s.wa)*a)

    def slope_steady_state(self, U):
        return (U / self.K_x)**(1/self.n) * self.stspm.A**(-self.m/self.n)

    def evolve(self, Nt=1_000_000, U=0.1e-3, dt=100, tol=1e-3, check_every=200):
        s = self.stspm
        z_spm = s.z.copy()
        slopes = np.empty_like(z_spm)
        for i in range(Nt):
            # area-based SPM
            z_spm[1:] += dt*U
            slopes[1:] = (z_spm[1:] - z_spm[:-1]) / s.dx;  slopes[0] = slopes[1]
            e_spm = self.K_x * s.A**self.m * np.maximum(slopes, 1e-8)**self.n * dt
            z_spm[1:] -= e_spm[1:]
            # ST-SPM
            s.z[1:] += U*dt
            e_st = s.calculate_time_step_incision(dt=dt, a=self.a, ke=self.ke_PA)
            s.z[1:] -= e_st[1:]

            if i % check_every == 0:
                mis = max(np.max(np.abs(e_spm[1:]/dt/U - 1)), np.max(np.abs(e_st[1:]/dt/U - 1)))
                print(f"step {i}/{Nt} | t={i*dt:.2e} yr | max|E/U-1| = {mis:.2e} (tol {tol:.0e})   ", end="\r")
                if mis < tol:
                    print(f"\nConverged after {i} steps ({i*dt:.2e} yr)");  break
        else:
            print(f"\nNot converged after {Nt} steps, max|E/U-1| = {mis:.2e}")
        self.z_spm, self.U, self.dt = z_spm, U, dt
        return mis < tol

    def test_equality(self, rtol=1e-2):
        z1, z2 = self.z_spm[1:], self.stspm.z[1:]
        err = np.max(np.abs(z1 - z2)) / np.max(np.abs(z1))
        print(f"max |z_SPM - z_STSPM| / max|z| = {err:.2e}")
        assert err < rtol, "area-based SPM and ST-SPM profiles differ"

    def test_steady_state(self, rtol=1e-2):
        # Steady state S = (U/K_x)^(1/n) * A^(-m/n), evaluated on the post-uplift profile
        s, U, dt = self.stspm, self.U, self.dt
        S_pred = self.slope_steady_state(U)
        for name, z in [("SPM", self.z_spm), ("ST-SPM", s.z)]:
            zp = z.copy();  zp[1:] += U*dt        # the balance holds before the erosion step
            S_num = np.empty_like(zp);  S_num[1:] = (zp[1:] - zp[:-1]) / s.dx
            err = np.abs(S_num[1:] / S_pred[1:] - 1)
            print(f"{name}: slope vs prediction: max rel. err = {err.max():.2e}, median = {np.median(err):.2e}")
            assert err.max() < rtol, f"{name} slopes deviate from steady-state prediction"
            z_pred = np.concatenate([[0.0], np.cumsum(S_pred[1:] * s.dx)]) + z[0]
            print(f"{name}: elevation vs integrated prediction: "
                  f"max rel. err = {np.max(np.abs(zp - z_pred))/np.max(z_pred):.2e}")

if __name__ == "__main__":
    
    from stspm import RiverSTSPM
    
    river = RiverSTSPM(grain_size_eff=0)
    river.set_constant_runoff(np.linspace(1, 2, len(river.x)))
    t = RiverSPMTest(river)
    t.evolve(Nt=400_000, U=1e-3, dt=100)
    t.test_equality()
    t.test_steady_state()