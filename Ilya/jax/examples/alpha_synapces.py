from custom_jax.neurosci import *
import data_preparation as dp

neurons_ids = [
    "7055857",
    "1805418",
    "14260575",
    "5835799",
    "10160250",
    "7840203",
    "5019924",
    "13986477",
    "10167078",
    "7982896",
    "4119387",
    "17591442",
    "4227544",
    "10495502",
    "8069478",
    "3913629",
    "11279244",
    "16846805",
    "8980589",
    "3664102",
]
neurons_ids = [int(i) for i in neurons_ids]
sc = dp.simulation_context_jax("20_del_this", neurons_ids)
csim = sc.get_jax_context()

num_nodes = csim['num_H']
num_syns = csim['num_syn']

def get_my_pipeline(csim, constants, dt = 0.1):
    integrate = get_euler_step(dt)
    HH = get_HH_pipeline(**constants)
    alpha = get_alpha_synapce_pipeline(csim['V_to_syn'], csim['syn_to_V'], **constants)
    cable = laplace_at_graph_symetric(csim['H_to_H'], 'V')
    @jax.jit
    def my_pipeline(state):
        s, ds = to_diff(state)
        s, ds = HH(s, ds)
        s, ds = cable(s, ds)
        ds['V'] += ds['V'].at[0].add((s['time'] > 20.0) * 30.0*(jnp.sin(s['time']/20.0) + 1.0)/2.0)
        s, ds = alpha(s, ds)
        s = integrate(s, ds)
        return s
    return my_pipeline

initials = {
    "V":jnp.ones((num_nodes, ), jnp.float32)*-65.0,
    "m":jnp.ones((num_nodes, ), jnp.float32)*0.0220,
    'n':jnp.ones((num_nodes, ), jnp.float32)*0.0773,
    'h':jnp.ones((num_nodes, ), jnp.float32)*0.9840,
    'alpha':jnp.ones((num_nodes, 2), jnp.float32)*0.0,
    "time":0.0
}

consts = {
    "C": jnp.ones((num_nodes, ), jnp.float32),      # Емкость мембраны (мкФ/см^2)
    "ENa": 50.0,   # Равновесный потенциал Na+ (мВ)
    "EK": -77.0,   # Равновесный потенциал K+ (мВ)
    "EL": -54.4,   # Равновесный потенциал утечки (мВ)
    "gNa": 120.0,  # Максимальная проводимость Na+ (мСм/см^2)
    "gK": 36.0,    # Максимальная проводимость K+ (мСм/см^2)
    "gL": 0.3,     # Проводимость утечки (мСм/см^2),
    "synaptic_weights":0.5,
    "alpha_syn_detector_treshold": 0.0,
    'E_rev':1.0,
    'G_max':10.0,
    "tau":2.0,
    "V_m":jnp.ones((num_syns, ), jnp.float32)*0.5
}

print('A stage')
my_pipeline = get_my_pipeline(csim, consts, 0.01)
print('pipline_constructed')
jsim = simulation(initials, my_pipeline, 100)
print('strted')
H = jsim.run(300)
print('ended')
import matplotlib.pyplot as plt
t, v = H['time'], H['V']
t, v = np.array(t), np.array(v)
plt.plot(t, v[:, 0])
plt.plot(t, v[:, 1])
plt.plot(t, v[:, 2])
plt.plot(t, v[:, 3])
plt.show()