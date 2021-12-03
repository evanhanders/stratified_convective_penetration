import numpy as np
import matplotlib.pyplot as plt

colors = ['purple', 'orange', 'green']

gamma = 5/3
R = 1
Cp = R*gamma/(gamma-1)
g = Cp
T_ad_z = g/Cp
grad_ad = (gamma - 1) / gamma
m_ad = 1/grad_ad - 1

mu = np.logspace(-3, 3, 1000)
P_vals = [1e-1, 1, 1e1]

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
ax1.axhline(grad_ad, c='k')
ax2.axhline(m_ad, c='k')

grad_rad_cz = (1 + mu) / (mu * R * g)
m_cz = 1/grad_rad_cz - 1
ax1.loglog(mu, grad_rad_cz, c='k', ls='--', label='CZ')
ax2.loglog(mu, m_cz, c='k', ls='--', label='CZ')
for i, P in enumerate(P_vals):
    ratio = 1 + (P*(1 + mu))**(-1)
    T_rad_z = T_ad_z/ratio
    grad_rad_rz = T_rad_z / (R*g)
    m_rz = 1/grad_rad_rz - 1
    ax1.loglog(mu, grad_rad_rz, label='RZ (P={})'.format(P), c=colors[i])
    ax2.loglog(mu, m_rz, label='RZ (P={})'.format(P), c=colors[i])




ax1.set_ylabel(r'$\nabla_{\rm{rad}}$')
ax1.set_xlabel(r'$\mu$')
ax1.legend()

ax2.set_ylabel(r'$m$')
ax2.set_xlabel(r'$\mu$')
ax2.set_yscale('linear')

fig.savefig('rz_nabla_or_m.png', dpi=300)




plt.show()
    
