import jax
from scripts.nj.utils import *

import jax.numpy as jnp
from jax import jit

def save_exp(x, max_value: float = 100.0):
    x = jnp.clip(x, a_max=max_value)
    return jnp.exp(x)

"""HH Sterratt, Graham, Gillies & Einevoll."""
SGGE_HH_channel_params = {
        "gNa": 0.12,
        "gK": 0.036,
        "gLeak": 0.0003,
        "eNa": 50.0,
        "eK": -77.0,
        "eLeak": -54.3,
    }


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


def _vtrap(x, y):
    return x / (save_exp(x / y) - 1.0)


# V = V_abs - E_rest, где E_rest обычно около -65 мВ. V=0 - это потенциал покоя.

@jit
def alpha_n(V):
    # Сингулярность при V = -10. Используем правило Лопиталя.
    V_plus_10 = V + 10.0
    
    # Замена для избежания деления на ноль:
    # Если V_plus_10 ~ 0, то alpha_n ~ 0.01 * 10 = 0.1
    # Это численный прием: при очень малых V_plus_10, exp(-V_plus_10 / 10) ~ 1 - V_plus_10 / 10.
    # Знаменатель: 1 - (1 - V_plus_10 / 10) = V_plus_10 / 10.
    # Функция: 0.01 * V_plus_10 / (V_plus_10 / 10) = 0.01 * 10 = 0.1
    
    return jnp.where(jnp.abs(V_plus_10) < 1e-4, 
                     0.1, 
                     0.01 * V_plus_10 / (1.0 - jnp.exp(-V_plus_10 / 10.0)))

@jit
def beta_n(V):
    return 0.125 * jnp.exp(-V / 80.0)


@jit
def alpha_m(V):
    # Сингулярность при V = -25. Используем правило Лопиталя.
    V_plus_25 = V + 25.0
    
    # При очень малых V_plus_25, alpha_m ~ 0.1 * 10 = 1.0
    
    return jnp.where(jnp.abs(V_plus_25) < 1e-4, 
                     1.0, 
                     0.1 * V_plus_25 / (1.0 - jnp.exp(-V_plus_25 / 10.0)))


@jit
def beta_m(V):
    return 4.0 * jnp.exp(-V / 18.0)


@jit
def alpha_h(V):
    return 0.07 * jnp.exp(-V / 20.0)


@jit
def beta_h(V):
    return 1.0 / (1.0 + jnp.exp(-(V + 30.0) / 10.0))


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

def generate_hh_channels_functions(C, ENa, EK, EL, gNa, gK, gL):
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
        return alpha_m(V) * (1 - m) - beta_m(V) * m

    @jax.jit
    def n_dynamic(V, n):
        return alpha_n(V) * (1 - n) - beta_n(V) * n

    @jax.jit
    def h_dynamic(V, h):
        return alpha_h(V) * (1 - h) - beta_h(V) * h

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

def get_alpha_synapce_pipeline(pre_synaptic, post_synaptic, tau, E_rev, G_max, V_m, C, alpha_syn_detector_treshold, synaptic_weights, treshold_interval = 0.01, *args, **kwargs):
    @jax.jit
    def u(alpha):
        da_dt = jnp.empty_like(alpha)
        da_dt.at[:, 0].set(-alpha[:, 0]/tau)
        da_dt.at[:, 1].set((alpha[:, 0] - alpha[:, 1])/tau)
        return da_dt
    
    @jax.jit
    def I(alpha, V_m):
        return G_max*alpha[:, 1]*(V_m - E_rev)
    

    @jax.jit
    def pipeline(state, ds_dt):
        ds_dt['alpha'] += u(state['alpha'])
        cin = C.at[post_synaptic[:, 1]].get()
        vim = V_m.at[post_synaptic[:, 1]].get()
        ain = state['alpha'].at[post_synaptic[:, 0]].get()
        ds_dt['V'] = ds_dt['V'].at[post_synaptic[:, 1]].add(-I(ain, vim)/cin)

        # instant changes
        v_ = state['V'].at[pre_synaptic[:, 0]].get()
        dv_dt_ = ds_dt['V'].at[pre_synaptic[:, 0]].get()

        delta_x = (jnp.abs(v_ - alpha_syn_detector_treshold) < treshold_interval) * (dv_dt_ > 0.0) * synaptic_weights
        state['alpha'] = state['alpha'].at[pre_synaptic[:, 1], 0].add(delta_x)
        return state, ds_dt
    return pipeline

def get_alpha_synapce_only_ds_dt_pipeline(pre_synaptic, post_synaptic, tau, E_rev, G_max, V_m, C, alpha_syn_detector_treshold, synaptic_weights, dt, treshold_interval = 0.01, *args, **kwargs):
    @jax.jit
    def u(alpha):
        da_dt = jnp.empty_like(alpha)
        da_dt.at[:, 0].set(-alpha[:, 0]/tau)
        da_dt.at[:, 1].set((alpha[:, 0] - alpha[:, 1])/tau)
        return da_dt
    
    @jax.jit
    def I(alpha, V_m):
        return G_max*alpha[:, 1]*(V_m - E_rev)
    
    @jax.jit
    def pipeline(state, ds_dt):
        ds_dt['alpha'] += u(state['alpha'])
        cin = C.at[post_synaptic[:, 1]].get()
        vim = V_m.at[post_synaptic[:, 1]].get()
        ain = state['alpha'].at[post_synaptic[:, 0]].get()
        ds_dt['V'] = ds_dt['V'].at[post_synaptic[:, 1]].add(-I(ain, vim)/cin)

        # instant changes
        v_ = state['V'].at[pre_synaptic[:, 0]].get()
        dv_dt_ = ds_dt['V'].at[pre_synaptic[:, 0]].get()

        delta_x = (jnp.abs(v_ - alpha_syn_detector_treshold) < treshold_interval) * (dv_dt_ > 0.0) * synaptic_weights
        #state['alpha'] = state['alpha'].at[pre_synaptic[:, 1], 1].add(delta_x)
        ds_dt['alpha'] = ds_dt['alpha'].at[pre_synaptic[:, 1], 0].add(delta_x/dt)
        return state, ds_dt
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

def get_HH_pipeline(C, ENa, EK, EL, gNa, gK, gL, *args, **kwargs):
    q = generate_hh_channels_functions(C, ENa, EK, EL, gNa, gK, gL)
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