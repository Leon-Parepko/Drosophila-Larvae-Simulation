from scripts.nj.neurosci import *
import scripts.data_preparation as dp

neurons_ids = [
    "7055857",
]
neurons_ids = [int(i) for i in neurons_ids]
sc = dp.simulation_context_jax("only_one_del_this", neurons_ids)
csim = sc.get_jax_context()

num_nodes = csim['num_H']

def get_my_pipeline(csim, constants, dt = 0.1):
    integrate = get_euler_step(dt)
    HH = get_HH_pipeline(**constants)
    cable = laplace_at_graph_symetric(csim['H_to_H'], 'V')
    @jax.jit
    def my_pipeline(state):
        s, ds = to_diff(state)
        s, ds = HH(s, ds)
        s, ds = cable(s, ds)
        ds['V'] += ds['V'].at[0].add((s['time'] > 20.0) * 30.0*(jnp.sin(s['time']/20.0) + 1.0)/2.0)
        s = integrate(s, ds)
        return s
    return my_pipeline

initials = {
    "V":jnp.ones((num_nodes, ), jnp.float32)*-65.0, 
    "m":jnp.ones((num_nodes, ), jnp.float32)*0.0220,
    'n':jnp.ones((num_nodes, ), jnp.float32)*0.0773,
    'h':jnp.ones((num_nodes, ), jnp.float32)*0.9840,
    "time":0.0
}

consts = {
    "C": 1.0,      # Емкость мембраны (мкФ/см^2)
    "ENa": 50.0,   # Равновесный потенциал Na+ (мВ)
    "EK": -77.0,   # Равновесный потенциал K+ (мВ)
    "EL": -54.4,   # Равновесный потенциал утечки (мВ)
    "gNa": 120.0,  # Максимальная проводимость Na+ (мСм/см^2)
    "gK": 36.0,    # Максимальная проводимость K+ (мСм/см^2)
    "gL": 0.3,     # Проводимость утечки (мСм/см^2),
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
plt.plot(t, v)
plt.show()