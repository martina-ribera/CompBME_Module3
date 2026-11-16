import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def Fact(L1, L2, Kd1, Kd2):
    """Fractional receptor activation with two agonists L1 and L2"""
    return (Kd2*L1+Kd1*L2) / (Kd1*Kd2 + Kd2*L1 + Kd1*L2)

# Optimized binding parameters from Task 1
Kd1 = 0.17       # GLP1 Kd (nM)
Kd2 = 0.077      # Optimized Cavalitide Kd (nM)
L1 = 0.015       # Fasting GLP1 concentration (nM)

# ===== choose product: 'Ozempic' or 'Wegovy' =====
drug = 'Cavilatide'

# product-specific parameters
products = {
    'Cavilatide':  dict(Favailable=0.01, #this is a pill so same dosage paramteres as wegovy assumed
                    ka=0.006,            # 1/h, absorption rate
                    D=21,               # mg
                    interval=24,       # h between doses (daily)
                    n_doses=100,
                    depot='gut (oral)'),
}
p = products[drug]
Favailable, ka, D = p['Favailable'], p['ka'], p['D']
interval, n_doses, depot = p['interval'], p['n_doses'], p['depot']

# shared model parameters
ke = 0.0039     # 1/h, elimination rate
V = 10.8        # L
MW = 4113.6     # g/mol
Ffree = 0.01    # rest of semaglutide bound to albumin

# simulation parameters
dt = 0.1        # h, Euler step size originally, now time interval for plotting


def ODEfunc(t, y):
    Asc, Cc = y
    dAsc = -ka*Asc            # Asc: depot amount (mg)
    dCc = ka*Asc/V - ke*Cc    # Cc: blood concentration (mg/L)
    return [dAsc, dCc]

def simulate(D, n_doses, interval, dt):
    """Repeated dosing every `interval` h; returns time (h), depot amount (mg),
    and blood concentration (nmol/L)."""
    t_out, Asc_out, Cc_out = [], [], []
    y = np.array([0.0, 0.0])
    n_steps = int(round(interval / dt))     ##STILL KEEP for new math# Euler steps per dosing interval 
    for i in range(n_doses):
        y[0] += Favailable*D                 # absorbed fraction of dose enters the depot
        t0 = i * interval
        t_seg = t0 + dt * np.arange(n_steps + 1)
        sol = solve_ivp(ODEfunc, [t_seg[0], t_seg[-1]], y, t_eval=t_seg) #given by vscode AI, this approximates using the new math method, which is more accurate than the Euler method
        y_seg = sol.y

        t_out.append(t_seg)
        Asc_out.append(y_seg[0])
        Cc_out.append(y_seg[1])
        y = y_seg[:, -1].copy()         # carry state into next interval
    t = np.concatenate(t_out)
    Asc_out = np.concatenate(Asc_out)
    Cc_out = np.concatenate(Cc_out)
    Cc_nM = Cc_out * 1e-3 / MW * 1e9
    return t, Asc_out, Cc_nM

# --- run ---
t, Asc, Cc = simulate(D, n_doses, interval, dt)
print("Final free Blood Concentration:", Ffree * Cc[-1])

Cc_free = Ffree * Cc

fraction_active = Fact(L1, Cc_free, Kd1, Kd2)


# ===== Results =====

print("Final free blood concentration:",
      Cc_free[-1], "nM")

print("Final fraction of active GLP1 receptors:",
      fraction_active[-1])

print("Final receptor activation:",
      fraction_active[-1] * 100, "%")


# ===== Plot PK and PD results =====

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))

# PK plot
ax1.plot(t / 24, Cc_free, color='blue')
ax1.set_xlabel("Time (days)")
ax1.set_ylabel("Free Cavalitide (nM)")
ax1.set_title("Cavalitide Blood Concentration")
ax1.grid(alpha=0.3)

# PD plot
ax2.plot(t / 24, fraction_active, color='red')
ax2.set_xlabel("Time (days)")
ax2.set_ylabel("Fraction of Active GLP1R Receptors")
ax2.set_title("GLP1 Receptor Activation Over 100 Days")
ax2.set_ylim(0, 1)
ax2.set_xlim(0, 100)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()