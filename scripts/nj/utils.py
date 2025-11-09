import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import scan, fori_loop


def laplace_at_graph_symetric(
    edges, key, scaling = None
):  # edges должны быть не ореинтированны и не повторятся
    q = jnp.array(edges, jnp.int32)
    static_sources = q[:, 0]
    static_targets = q[:, 1]
    if scaling is None:
        def graph_evolution_fn_without_scaling(X: jnp.ndarray, dx_dt) -> jnp.ndarray:
            potential_diff = (
                X[key].at[static_targets].get() - X[key].at[static_sources].get()
            )  # возможно нужно нормировать с учетом количества соседей
            dx_dt[key] = dx_dt[key].at[static_sources].add(potential_diff)
            dx_dt[key] = dx_dt[key].at[static_targets].add(-potential_diff)
            return X, dx_dt
        graph_evolution_fn = graph_evolution_fn_without_scaling
    else:
        def graph_evolution_fn_with_scaling(X: jnp.ndarray, dx_dt) -> jnp.ndarray:
            potential_diff = (
                X[key].at[static_targets].get() - X[key].at[static_sources].get()
            )  # возможно нужно нормировать с учетом количества соседей
            dx_dt[key] = dx_dt[key].at[static_sources].add(potential_diff*scaling.at[static_sources].get())
            dx_dt[key] = dx_dt[key].at[static_targets].add(-potential_diff*scaling.at[static_targets].get())
            return X, dx_dt
        graph_evolution_fn = graph_evolution_fn_with_scaling

    return jax.jit(graph_evolution_fn)

#TODO ? нам это вообще нужно ?
def laplace_at_graph_oriented(
    edges, key, scaling = None
):
    q = jnp.array(edges, jnp.int32)
    static_sources = q[:, 0]
    static_targets = q[:, 1]


    if scaling is None:
        def graph_evolution_fn_without_scaling(X: jnp.ndarray, dx_dt) -> jnp.ndarray:
            potential_diff = (
                X[key].at[static_targets].get() - X[key].at[static_sources].get()
            )  # возможно нужно нормировать с учетом количества соседей
            dx_dt[key] = dx_dt[key].at[static_sources].add(potential_diff)
            dx_dt[key] = dx_dt[key].at[static_targets].add(-potential_diff)
            return X, dx_dt
        graph_evolution_fn = graph_evolution_fn_without_scaling
        
    else:
        def graph_evolution_fn_with_scaling(X: jnp.ndarray, dx_dt) -> jnp.ndarray:
            potential_diff = (
                X[key].at[static_targets].get() - X[key].at[static_sources].get()
            )  # возможно нужно нормировать с учетом количества соседей
            dx_dt[key] = dx_dt[key].at[static_sources].add(potential_diff*scaling[0])
            dx_dt[key] = dx_dt[key].at[static_targets].add(-potential_diff*scaling[1])
            return X, dx_dt
        graph_evolution_fn = graph_evolution_fn_with_scaling

    return jax.jit(graph_evolution_fn)

def get_euler_step(dt):
    @jax.jit
    def euler_step(state, v):
        return jax.tree_util.tree_map(lambda x, y: x+ y*dt, state, v)
    return euler_step

def get_scan_integration_function(step_foo, inside_iterations):
    @jax.jit
    def step_fn(i, carry):
        state = step_foo(carry)
        return state

    @jax.jit
    def inner_loop_step(start_state, num_steps):
        state = fori_loop(0, num_steps, step_fn, start_state)
        return state

    @jax.jit
    def scan_step(carry, unused_input):
        """(final_x1, final_x2, ...), (evolution_x1_jnp, evolution_x2_jnp, ...)"""

        state = inner_loop_step(
            start_state=carry, num_steps=inside_iterations
        )
        return state, state
    
    return scan_step

@jax.jit
def to_diff(state):
    d = jax.tree_util.tree_map(jnp.zeros_like, state)
    d['time'] = 1.0
    return state, d

class simulation:
    def __init__(self, initials:dict, pipeline_fun, inside_iterations):
        self.state = initials
        self.pipeline = pipeline_fun
        self.inside_iterations =inside_iterations
        self.get_integraton_process()

    def get_integraton_process(self):
        pipline = self.pipeline
        inside_iterations = self.inside_iterations
        self.scan_step = get_scan_integration_function(pipline, inside_iterations)
        return self.scan_step
    
    def run(self, iterations):
        iters = jnp.arange(0, iterations)
        self.state, self.history = scan(
            f=self.scan_step,
            init=self.state,
            xs=iters
        )
        return self.history