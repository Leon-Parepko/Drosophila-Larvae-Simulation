from collections import namedtuple
import jax
import numpy as np
from scripts.new_nj.utils import BASICSEP
from scripts.nj.utils import *
import jax.numpy as jnp
from jax import jit
from types import SimpleNamespace


def save_exp(x, max_value: float = 100.0):
    x = jnp.clip(x, a_min=-jnp.inf, a_max=max_value)
    return jnp.exp(x)

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

    return SimpleNamespace(**{
        "INa": INa,
        "IK": IK,
        "Ileak": Ileak,
        "V_dynamic": V_dynamic,
        "m_dynamic": m_dynamic,
        "n_dynamic": n_dynamic,
        "h_dynamic": h_dynamic,
    })



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


def to_dict(x):
    pass

def create_initial_array(initial, shape):
    if shape is None:
        return np.ones(shape, np.float32)*initial
    assert initial.shape == shape
    return initial

class complex_pipeline:
    def __init__(self, pipeline_former, path:str, name:str, consts:dict = None, consts_shapes:dict = None, variables_shape:dict = None, variables_initials = None, sep = BASICSEP):
        self.path = path
        self.name = name
        self.__total_path = path + sep + name
        if consts is not None:
            self.consts = {
                (self.__total_path + sep + k):create_initial_array(c, consts_shapes.get(k, None))
                for k, c in consts.items()
            }

        if variables_initials is not None:
            self.variables_shape = {(self.__total_path + sep + k):s for k, s in variables_shape.items()}
            self.variables_initials = {
                    (self.__total_path + sep + k):create_initial_array(v, variables_shape.get(k, None))
                    for k, v in variables_initials.items()
                }
        self.to_form = pipeline_former

    def process_basics(self, variables_initials, consts):
        if consts is None:
            consts = dict()
        consts = to_dict(consts)
        consts = self.BASIC_HH_CONSTS | consts

        if variables_initials is None:
            variables_initials = dict()
        variables_initials = to_dict(variables_initials)
        variables_initials = self.BASIC_INITIALS | variables_initials
        return variables_initials, consts
    
    def form_pipeline(self):
        self.__call__ = self.__to_form()
        return super().form_pipeline()

    def form_state(self):
        return dict(**self.variables_initials)

    def form_consts(self):
        return dict(**self.consts)


class HH_pipeline(complex_pipeline):
    BASIC_CONSTS = {
        "C": 1.0,# Емкость мембраны (мкФ/см^2)
        "ENa": 50.0,   # Равновесный потенциал Na+ (мВ)
        "EK": -77.0,   # Равновесный потенциал K+ (мВ)
        "EL": -54.4,   # Равновесный потенциал утечки (мВ)
        "gNa": 120.0,  # Максимальная проводимость Na+ (мСм/см^2)
        "gK": 36.0,    # Максимальная проводимость K+ (мСм/см^2)
        "gL": 0.3,     # Проводимость утечки (мСм/см^2),
    }
    BASIC_INITIALS =  {
        "V": -60.0,
        "m": 0.05529518,
        'n': 0.32336363,
        'h': 0.58303326,
    }
    def __init__(self, path, name, num, variables_initials=None, consts = None, sep=BASICSEP):
        self.__total_path = path + sep + name
        variables_initials, consts = self.process_basics(variables_initials, consts)


        consts_shapes = {k:(num, ) for k, c in consts.items()}
        variables_shape = {k:(num, ) for k, c in variables_initials.items()}
        def to_form():
            q = generate_hh_channels_functions_SGGE(**consts)
            dv = q.V_dynamic
            dm = q.m_dynamic
            dn = q.n_dynamic
            dh = q.h_dynamic
            self.V_key = V_key = self.__total_path + sep + 'V'
            self.m_key = m_key = self.__total_path + sep + 'm'
            self.n_key = n_key = self.__total_path + sep + 'n'
            self.h_key = h_key = self.__total_path + sep + 'h'

            @jax.jit
            def pipeline(state, ds_dt):
                ds_dt[V_key] += dv(state[V_key], state[m_key], state[n_key], state[h_key])
                ds_dt[m_key] += dm(state[V_key], state[m_key])
                ds_dt[n_key] += dn(state[V_key], state[n_key])
                ds_dt[h_key] += dh(state[V_key], state[h_key])
                return state, ds_dt

        super().__init__(path, to_form,  name, consts, consts_shapes, variables_shape, variables_initials, sep)


class alpha_synapce(complex_pipeline):
    BASIC_CONSTS = {
        "tau":1.25,
        "E_rev":0.0,
        "G_max":1.0,
        "V_m":1.0,
        "C":1.0,
        "detector_treshold":0.0,
        "synaptic_weights":0.5,
    }
    BASIC_INITIALS =  {
        "V": -60.0,
        "m": 0.05529518,
        'n': 0.32336363,
        'h': 0.58303326,
    }

    def __init__(self, path, name, num_syn, pre_key, post_key,
                pre_synaptic, post_synaptic, delay_ms, dt,
                consts = None, variables_initials=None, treshold_interval = 0.001, sep=BASICSEP):
        self.__total_path = path + sep + name
        variables_initials, consts = self.process_basics(variables_initials, consts)

        #tau, E_rev, G_max, V_m, C, detector_treshold, synaptic_weights, dt, treshold_interval

        x_name = self.__total_path + sep + 'x'
        y_name = self.__total_path + sep + 'y'
        consts_shapes = {k:(1, ) for k, c in consts.items()}

        synaptic_weights = consts['synaptic_weights']
        tau = consts['tau']
        E_rev = consts['E_rev']
        G_max = consts['G_max']
        C = consts['C']
        detector_treshold = consts['detector_treshold']

        delay_steps = int(delay_ms / dt)

        delay_name = self.__total_path + sep + 'delay_y'
        name_path = self.__total_path
        variables_shape = {
            'x':(num_syn, ),
            'y':(num_syn, ),
            'delay_y':(num_syn, delay_steps),
        }

        def to_form():
            @jax.jit
            def u(x, y):
                x = -x/tau
                y = (x - y)/tau
                return x, y

            @jax.jit
            def I(y, V_m):
                return G_max*y*(V_m - E_rev)
            rbd = delayed_ring(delay_name, y_name, dt)

            @jax.jit
            def pipeline(state, ds_dt):
                state = rbd(state)
                x, y = u(state[name_path])
                ds_dt[x_name] += x
                ds_dt[y_name] += y


                # post synaptic
                cin = C.at[post_synaptic[:, 1]].get()
                vim = state[post_key].at[post_synaptic[:, 1]].get()
                buffer_ind = ring_get(state, delay_name, dt)
                y = state[delay_name].at[post_synaptic[:, 0], buffer_ind].get()
                ds_dt[post_key] = ds_dt[post_key].at[post_synaptic[:, 1]].add(-I(y, vim)/cin)


                # instant changes (pre synaptic)
                v_ = state[pre_key].at[pre_synaptic[:, 0]].get()
                dv_dt_ = ds_dt[pre_key].at[pre_synaptic[:, 0]].get()

                is_near_threshold = jnp.abs(
                    v_ - detector_treshold) < treshold_interval
                is_rising = dv_dt_ > 0.0
                delta_x = is_near_threshold * is_rising * synaptic_weights
                state[y_name] = state[y_name].at[pre_synaptic[:, 1]].add(delta_x)
                return state, ds_dt
            return pipeline
        super().__init__(path, to_form, name, consts, consts_shapes, variables_shape, variables_initials, sep)