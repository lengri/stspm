import numpy as np
from scipy.stats import invgamma
from scipy.optimize import minimize

def hack_A(l, c, h, Amin):
    """Area at distance l from the channel head (A = Amin there). Hack's law L = c A^h."""
    return (Amin**h + l/c)**(1/h)

def fit_hack(x, A):
    """x: distance from outlet (ascending), A: area; data may be truncated upstream."""
    x = np.asarray(x, float);  A = np.asarray(A, float)
    i_head = np.argmax(x);  Ah = A[i_head];  L = x[i_head]
    l = L - x                                            # distance from the (truncated) data head
    def mf(par):
        logc, h = par
        return np.sum((np.log(hack_A(l, 10**logc, h, Ah)) - np.log(A))**2)
    h0 = 0.6;  c0 = L/(A[0]**h0 - Ah**h0)
    res = minimize(mf, [np.log10(c0), h0], method="Powell", bounds=[(-12, 6), (0.05, 2)])
    return 10**res.x[0], res.x[1], Ah, L                 # c, h, area at data head, data length

def hack_full(c, h, Ah, L_data, Amin=1e5, dx=1e3):
    """Extend the fitted law upstream to A = Amin. x = 0 at the outlet."""
    L_ext = c*(Ah**h - Amin**h)                          # extra channel length above the data head
    L_tot = L_data + L_ext
    x = np.arange(0, L_tot + dx, dx)
    return x, hack_A(L_tot - x, c, h, Amin), L_ext
    

def sec2yr(t):
    # 60*60*24*365.25
    return t*31557600

def invgamma_record(mean_x, std_x, edges):
    """mean_x, std_x: (nx,) target mean and std of runoff [m/yr].
    edges: (nb+1,) bin edges [m/yr]. Returns bin centres (nb,) and masses (nx, nb)."""
    mean_x = np.asarray(mean_x, float);  std_x = np.asarray(std_x, float)
    
    if np.any(std_x <= 0):
        raise ValueError("std must be > 0; use set_constant_runoff for CV = 0")

    cv2 = (std_x / mean_x)**2
    a = 2.0 + 1.0 / cv2                      # shape; mean = b/(a-1), var = b^2/((a-1)^2 (a-2))
    b = mean_x * (a - 1.0)                   # scale
    cdf = invgamma.cdf(edges[None, :], a[:, None], scale=b[:, None])
    w = np.diff(cdf, axis=1)                 # exact probability mass per bin
    rf_vals = np.sqrt(edges[1:] * edges[:-1])    # geometric bin centres
    return rf_vals, w

def identify_ke_equal_slope_no_thresh(rSTSPM, U, a=1, target_index=None):
    """ke (m/yr/Pa^a) for a threshold model rSTSPM (runoff record already set) such that
    the steady-state slope at node `target_index` equals s_target, the equilibrium slope
    of the threshold case at the same node."""
    if target_index is None:
        # Find id with A = Amax/2
        target_index = np.argmin(np.abs(rSTSPM.A.max()/2-rSTSPM.A))
        
    i = target_index
    s_target = rSTSPM.calculate_slope()[i]
    
    PA = 1e3 * sec2yr(1)**2
    rfmean = rSTSPM.calculate_average_runoff()[i]
    expo1 = rSTSPM.alpha * (1 - rSTSPM.ws)
    expo2 = rSTSPM.alpha * (rSTSPM.ws - rSTSPM.wa)
    expo3 = rSTSPM.alpha * (1 - rSTSPM.wa)

    fact   = rSTSPM.kw**(-rSTSPM.alpha) * rSTSPM.kt * rfmean**expo2 * rSTSPM.A[i]**expo3 * s_target**rSTSPM.beta
    tau_Pa = fact * rSTSPM.rf_vals[i]**expo1 / PA                 # Pa, one value per runoff bin
    return U / np.sum(rSTSPM.rf_w[i] * tau_Pa**a)                 # ke * <tau^a> = U

def set_case_runoff(r, Pmean_x, CV_x, edges, cv_min=0.2):
    """CV = 0 everywhere -> constant runoff; otherwise inverse gamma record."""
    if np.all(CV_x == 0):
        r.set_constant_runoff(Pmean_x)
    elif np.any(CV_x < cv_min):
        raise ValueError(f"CV must be 0 everywhere or >= {cv_min} at every node "
                         f"(narrow distributions are not resolved by the bin grid)")
    else:
        v, w = invgamma_record(Pmean_x, CV_x * Pmean_x, edges)
        r.set_runoff_record(v, w)
    # Truncation at the upper bin edge should not change the mean noticeably
    assert np.allclose(r.calculate_average_runoff(), Pmean_x, rtol=2e-2), "mean runoff off target"

def chi_transform(d, w):
    """chi(x) = integral of w dx from the outlet; trapezoid for irregular node spacing."""
    return np.r_[0.0, np.cumsum(0.5*(w[1:] + w[:-1])*np.diff(d))]