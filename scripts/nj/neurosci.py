'''
на чем я остановился:
на интеграции задержки и alpha синапсов.
На данный момент я хочу сделать задержку в синапсах по умолчанию
Потом я хочу переписать все в терминах классов, что бы хранить информацию о начальных значениях
'''

import jax
from scripts.nj.utils import *
import jax.numpy as jnp
from jax import jit


def save_exp(x, max_value: float = 100.0):
    x = jnp.clip(x, a_min=-jnp.inf, a_max=max_value)
    return jnp.exp(x)

"""HH Sterratt, Graham, Gillies & Einevoll."""
SGGE_HH_channel_params = {
        "gNa": 0.12, # mS/cm^2
        "gK": 0.036, # mS/cm^2
        "gLeak": 0.0003, # mS/cm^2
        "eNa": 50.0, #mV
        "eK": -77.0, #mV
        "eLeak": -54.3, #mV
    }
# C в µF/cm^2 (µ - микро)

def m_gate(v):
    alpha = 0.1 * _vtrap(-(v + 40), 10)
    beta = 4.0 * save_exp(-(v + 65) / 18)
    return alpha, beta

def h_gate(v):
    alpha = 0.07 * save_exp(-(v + 65) / 20)
    beta = 1.0 / (save_exp(-(v + 35) / 10) + 1)
    return alpha, beta

def n_gate(v):
    alpha = 0.01 * _vtrap(-(v + 55), 10)
    beta = 0.125 * save_exp(-(v + 65) / 80)
    return alpha, beta


def _vtrap(x, y, epsilon=1e-12):
    return x / (save_exp(x / y) - 1.0 + epsilon)


def generate_hh_channels_functions_SGGE(C, ENa, EK, EL, gNa, gK, gL):
    """HH from Sterratt, Graham, Gillies & Einevoll."""
    @jax.jit
    def INa(V, m, h):
        return gNa * h * m**3 * (V - ENa)

    @jax.jit
    def IK(V, n):
        return gK * n**4 * (V - EK)

    @jax.jit
    def Ileak(V):
        return gL * (V - EL)

    @jax.jit
    def m_dynamic(V, m):
        alpha, beta = m_gate(V)
        return alpha * (1 - m) - beta * m

    @jax.jit
    def n_dynamic(V, n):
        alpha, beta = n_gate(V)
        return alpha * (1 - n) - beta * n

    @jax.jit
    def h_dynamic(V, h):
        alpha, beta = h_gate(V)
        return alpha * (1 - h) - beta * h

    @jax.jit
    def V_dynamic(V, m, n, h):
        return -(INa(V, m, h) + IK(V, n) + Ileak(V))/C

    return {
        "INa": INa,
        "IK": IK,
        "Ileak": Ileak,
        "V_dynamic":V_dynamic,
        "m_dynamic": m_dynamic,
        "n_dynamic": n_dynamic,
        "h_dynamic": h_dynamic,
    }

def get_alpha_synapce_pipeline(pre_synaptic, post_synaptic, tau, E_rev, G_max, C, alpha_syn_detector_treshold,
                                synaptic_weights, dt, treshold_interval = 0.1, pre_key = 'V', post_key = 'V', name = 'alpha',
                                delay_name = 'alpha_delay', can_spike_name = "alpha_timer", detection_lower_speed = 1,
                                  *args, **kwargs):
    @jax.jit
    def u(alpha):
        da_dt = jnp.empty_like(alpha)
        da_dt = da_dt.at[:, 0].set(-alpha[:, 0]/tau)
        da_dt = da_dt.at[:, 1].set((alpha[:, 0] - alpha[:, 1])/tau)
        return da_dt
    
    @jax.jit
    def I(alpha, V_m):
        return G_max*alpha[:, 1]*(V_m - E_rev)
    rbd = delayed_ring(delay_name, name, dt)
    
    @jax.jit
    def pipeline(state, ds_dt):
        state = rbd(state)
        ds_dt[name] = ds_dt[name] + u(state[name])

        cin = C.at[post_synaptic[:, 1]].get()
        vim = state[post_key].at[post_synaptic[:, 1]].get()
        buffer_ind = ring_get(state, delay_name, dt)
        ain = state[delay_name].at[post_synaptic[:, 0], buffer_ind].get()
        #ain = state[name].at[post_synaptic[:, 0]].get()
        ds_dt[post_key] = ds_dt[post_key].at[post_synaptic[:, 1]].add(-I(ain, vim)/cin)


        # instant changes and timer
        v_ = state[pre_key].at[pre_synaptic[:, 0]].get()
        dv_dt_ = ds_dt[pre_key].at[pre_synaptic[:, 0]].get()
        is_near_threshold = jnp.abs(v_ - alpha_syn_detector_treshold) < treshold_interval
        is_rising = dv_dt_ > 0.0

        # timer changes
        able_spike = (state[can_spike_name] < 0.1)
        ds_dt[can_spike_name] = -detection_lower_speed*(1 - able_spike)
        state[can_spike_name] = is_near_threshold * is_rising * able_spike + state[can_spike_name]*(1 - able_spike)

        # instant changes
        delta_x = is_near_threshold * is_rising * synaptic_weights
        state[name] = state[name].at[pre_synaptic[:, 1], 0].add(delta_x)*able_spike + (1 - able_spike)*state[name]
        return state, ds_dt
    return pipeline


def ring_get(state, key, dt):
    current_step_idx = jnp.floor(state['time'] / dt).astype(jnp.int32)
    buffer_ind = current_step_idx % state[key].shape[1]
    return buffer_ind

def delayed_ring(name, from_key, dt):
    @jax.jit
    def pipeline(state):
        buffer_ind = ring_get(state, name, dt)
        state[name] = state[name].at[:, buffer_ind].set(state[from_key])
        return state
    return pipeline

def get_HH_pipeline_SGGE(C, ENa, EK, EL, gNa, gK, gL, *args, **kwargs):
    q = generate_hh_channels_functions_SGGE(C, ENa, EK, EL, gNa, gK, gL)
    dv = q['V_dynamic']
    dm = q['m_dynamic']
    dn = q['n_dynamic']
    dh = q['h_dynamic']
    @jax.jit
    def pipeline(state, ds_dt):
        ds_dt['V'] += dv(state['V'], state['m'], state['n'], state['h'])
        ds_dt['m'] += dm(state['V'], state['m'])
        ds_dt['n'] += dn(state['V'], state['n'])
        ds_dt['h'] += dh(state['V'], state['h'])
        return state, ds_dt

    return pipeline