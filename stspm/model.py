import numpy as np

from .helpers import sec2yr

class RiverSTSPM:
    
    def __init__(
        self,
        x: None|np.ndarray = None,
        A: None|np.ndarray = None,
        kf_friction_coeff: float = 0.15,
        r_friction_exp: float = 0.25,
        alpha: float = 0.6,
        beta: float = 0.7,
        rho_water: float = 1e6,
        rho_sed: float = 2.7e6,
        grain_size_eff: float = 0.05,
        manning_n: float = 0.03/sec2yr(1),
        kw_width_coeff: float = 0.0008, # Rough calibration for wa = 0.5 on Himalayan rivers
        omega_s_station: float = 0.35,
        omega_a_downstream: float = 0.5,
        gravity_accel: float = 9.81*(sec2yr(1)**2),
        z_init: None|np.ndarray = None,
    ):
        
        """
        Parameters
        ----------
        
        x: np.ndarray
            Distance of river nodes from outlet [m]. Default is 5 km long river with 
            50 m node spacing.
        A: np.ndarray
            Drainage area of each channel node. Default is calculated following Hack's Law
            with h=0.6 and 
        kf_friction_coeff: float
            Friction coefficient of the slope-dependent Shields number law of Lamb et al. (2009):
            tau_c* = kf_friction_coeff * S ** r_friction_exp.
            Default kf_friction_coeff = 0.15
        r_friction_exp: float
            Friction exponent of the slope-dependent Shields number law of Lamb et al. (2009):
            tau_c* = kf_friction_coeff * S ** r_friction_exp.
            Default r_friction_exp = 0.25
        alpha: float
            Exponent of the unit-discharge (discharge per unit width) term in the hydraulic friction
            law,
                tau = k_t * (Q / W(Q))**alpha * S**beta
        beta: float
            Exponent of the slope term in the hydra
        theta: float
            Slope-exponent of Flint's Law S = k_s * A^-theta
        rho_water: float
            Density of water, 1e6 g/m^3
        rho_sed: float
            Density of bedload sediment, 2.7e6 g/m^3
        grain_size_eff: float
            The effective grain size, here set to 0.05 m following Scherler et al. (2017). Note
            that setting the grain size to 0 results in a threshold-free incision model.
        manning_n: float
            Manning friction coefficient, 0.03. Note that t
        kw_width_coeff: float
            Width-scaling coefficient. Default is 1e-2. This results in a ~30 m wide channel 
            for default options (5 km long channel)
        omega_s_station: float
            Width-exponent of the at-a-station variation of width with discharge. Default is 0.34
            following Turowski et al. (2008)
        omega_a_downstream: float
            Width-exponent of the downstream variation of channel width. Default is 0.5 according to 
            Lague et al. (2014)
        gravity_accel: float
            Gravitational acceleration constant in m/year^2. Default is 9.81 m/s^2.
        
        """
        
        if x is None:
            x = np.arange(0, 5e3+50, 50)
        if z_init is None:
            z_init = 1e-2*x
        # if A is None, define A using a modified Hack's Law where
        # A = Amin * (1+(xmax-x)/Lc)^(1/h)
        # h is Hack's exponent and Lc is a characteristic length scale
        # somewhat reflecting the role of C. Here, Lc = 280 calibrated
        # by fitting A-x data of Himalayan rivers.
        if A is None:
            # min A = 1e5 m^2; 
            A = 1e5 * (1+(x.max()-x)/280)**(1/0.6)
        

        self.x = x
        self.z = z_init
        self.A = A
        self.dx = self.x[1]-self.x[0]
        
        # TODO: Ensure that all 3 params have same shape
        
        # Save fixed parameters
        self.kf = kf_friction_coeff
        self.r = r_friction_exp
        self.alpha = alpha
        self.beta = beta
        self.rho_s = rho_sed
        self.rho_w = rho_water
        self.N = manning_n
        self.kw = kw_width_coeff
        self.ws = omega_s_station
        self.wa = omega_a_downstream
        self.Deff = grain_size_eff
        self.kt = gravity_accel*rho_water*manning_n**(3/5)
        self.kt_no_g = rho_water*manning_n**(3/5)
        self.g = gravity_accel
    
    def set_runoff_record(self, rf_values, rf_weights):
        """
        rf_weights: probability mass of each runoff value (sums to 1 per node).
        This function expects each runoff distribution to contain upstream-averaged values
        for that specific node.
        
        """
        rf_values  = np.asarray(rf_values, float);  rf_weights = np.asarray(rf_weights, float)
        nx, nb = len(self.x), rf_values.shape[-1]
        self.rf_vals = np.broadcast_to(rf_values,  (nx, nb)).copy()
        w            = np.broadcast_to(rf_weights, (nx, nb)).copy()
        self.rf_w    = w / w.sum(axis=1, keepdims=True)

    def set_constant_runoff(self, R):
        """R: scalar (same everywhere) or array of shape (nx,) with one runoff value per node [m/yr]."""
        R = np.broadcast_to(np.asarray(R, float), (len(self.x),))
        self.set_runoff_record(R[:, None], np.ones((len(self.x), 1)))

    def calculate_average_runoff(self):
        return (self.rf_vals * self.rf_w).sum(axis=1)
    
    def calculate_runoff_std(self):
        mean = self.calculate_average_runoff()
        var  = (self.rf_w * (self.rf_vals - mean[:, None])**2).sum(axis=1)
        return np.sqrt(var)

    def calculate_runoff_cv(self):
        return self.calculate_runoff_std() / self.calculate_average_runoff()

    def calculate_slope(self):
        
        slopes = np.empty_like(self.z)
        slopes[1:] = (self.z[1:] - self.z[:-1]) / self.dx
        slopes[0]  = slopes[1]
        
        return slopes
    
    def calculate_critical_runoff(self):
        s = np.maximum(self.calculate_slope(), 1e-8)
        rfmean = self.calculate_average_runoff()
        expo1 = self.alpha*(1-self.ws)
        expo2 = self.alpha*(self.ws-self.wa)
        expo3 = self.alpha*(1-self.wa)
        num = self.kf * s**(self.r-self.beta) * self.kw**self.alpha * (self.rho_s-self.rho_w) * self.Deff
        den = self.kt_no_g * rfmean**expo2 * self.A**expo3
        return (num/den)**(1/expo1)

    def calculate_fractional_moment(self, a=1, p=None):
        """
        Fractional moment of the runoff distribution at each node,
            Sigma_R = int p(R) R^p dR = sum_k w_k R_k^p,
        with p = alpha * (1 - omega_s) * a unless given explicitly.
        Runoff in m/yr, i.e. the same convention as the SigmaR column in the river CSVs.
        p = 1 gives the mean runoff.

        Returns
        -------
        np.ndarray, shape (n_nodes,)
        """
        if p is None:
            p = self.alpha*(1 - self.ws)*a
        # histogram data: probability-mass weights, so plain sums (no trapezoid)
        return (self.rf_w * self.rf_vals**p).sum(axis=1)

    def calculate_threshold_excess_moment(self, a=1, p=None, Rc=None):
        """
        Threshold-excess moment of the runoff distribution at each node,
            G_a(Rc) = int_{Rc}^inf p(R) [ (R/Rc)^p - 1 ] dR
                    = sum_k w_k [ (R_k/Rc)^p - 1 ]_+ ,
        with p = alpha * (1 - omega_s) * a unless given explicitly. Only runoff bins
        above the critical runoff Rc contribute.

        Parameters
        ----------
        a : float
            Incision exponent (sets p).
        p : float, optional
            Override the exponent of the moment.
        Rc : np.ndarray (n_nodes,) or float, optional
            Critical runoff [m/yr]. By default calculate_critical_runoff() is used, which
            depends on the current slope.

        Returns
        -------
        np.ndarray, shape (n_nodes,). Dimensionless. Where Rc = 0 (Deff = 0) the value is
        inf; use calculate_fractional_moment for the threshold-free limit.
        """
        if p is None:
            p = self.alpha*(1 - self.ws)*a
        if Rc is None:
            Rc = self.calculate_critical_runoff()
        Rc = np.broadcast_to(np.asarray(Rc, dtype=float), (self.rf_vals.shape[0],))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ratio = self.rf_vals / Rc[:, None]
            excess = np.where(ratio > 1.0, ratio**p - 1.0, 0.0)   # [.]_+ : only R > Rc
        return (self.rf_w * excess).sum(axis=1)

    def calculate_variability_factor(self, a=1, p=None):
        """
        Dimensionless variability factor Phi = Sigma_R / Rmean^p = E[(R/Rmean)^p].
        Equals 1 for constant runoff (or for p = 1), < 1 for p < 1 with variable runoff.
        Independent of the time unit used for R.
        """
        if p is None:
            p = self.alpha*(1 - self.ws)*a
        return self.calculate_fractional_moment(a=a, p=p) / self.calculate_average_runoff()**p

    def calculate_time_step_incision(
        self,
        dt = 100, # Years
        a: float = 1, # between 1 and 3.5,function of dominant erosion process
        ke: float = 1e-2 # incision scaling of the instantaneous erosion equation
    ):
        
        """
        Provide ke in m/yr/Pa, normal rates are around 1e-2 to 1e-4.
        """
        
        self.ke = ke
        self.a = a
        
        s = np.maximum(self.calculate_slope(), 1e-8)
        rfmean = self.calculate_average_runoff()
        
        # Calculate tau for each node and each runoff bin
        
        # Calculate constant exponents
        expo1 = self.alpha*(1-self.ws)
        expo2 = self.alpha*(self.ws-self.wa)
        expo3 = self.alpha - self.alpha*self.wa
        
        fact = self.kw**(-self.alpha)*self.kt*rfmean**expo2*self.A**expo3*s**self.beta
        
        # Calculate critical shear stress for each x
        tau_cx = (self.kf * s**self.r*self.g*(self.rho_s-self.rho_w)*self.Deff)**a
        
        # Calculate effective shear stress and erosion rate for each rf_vals bin center
        # (tau**a - tau_c**a). If negative, the erosion is set to 0.
        
        # Note that the conversion of (tau-tau_c)**a to (tau**a-tau_c**a) follows
        # Lague (2014), Scherler et al. (2017) who use the latter form first (?) proposed in
        # Tucker (2004). There is no reason to prefer on form over the other for this 
        # numerical model, but using the second form makes the results comparable to some
        # analytical solutions.
        
        PA = 1e3 * sec2yr(1)**2                                   # g m^-1 yr^-2 per Pa

        # Calculate shear stress in Pa
        tau_Pa  = fact[:, None] * self.rf_vals**expo1 / PA        # Pa
        tauc_Pa = self.kf * s**self.r * self.g * (self.rho_s - self.rho_w) * self.Deff / PA   # Pa

        e_per_x_rf = ke * np.clip(tau_Pa**a - tauc_Pa[:, None]**a, 0, None)   # ke in m/yr/Pa^a
        
        # Total erosion rate
        e_in_dt = (self.rf_w * e_per_x_rf).sum(axis=1) * dt
        
        return e_in_dt  
        