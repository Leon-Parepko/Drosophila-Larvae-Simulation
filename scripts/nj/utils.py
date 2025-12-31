import jax
import jax.numpy as jnp
from jax.lax import scan, fori_loop
from jax.scipy.sparse.linalg import cg

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

def get_backward_euler_linear_step(linear_function, key, dt):
    @jax.jit
    def backward_euler_step(state):
        b = state[key]
        def system_operator(x):
            state = {**state, key: x}
            dx_dt = {k: jnp.zeros_like(v) for k, v in state.items()}
            
            _, updated_dx_dt = linear_function(state, dx_dt)
            
            # (I - dt * L) @ x
            return x - dt * updated_dx_dt[key]
        new_val, _ = cg(system_operator, b, x0=b, tol=1e-5)
        state[key] = new_val
            
        return state

    return backward_euler_step

def get_euler_step(v_func, dt):
    @jax.jit
    def euler_step(state):
        v, dv = v_func(state)
        return jax.tree_util.tree_map(lambda x, y: x+ y*dt, v, dv)
    return euler_step

def get_runge_kutta_step(v_func, dt):
    @jax.jit
    def runge_kutta_step(state):
        S_a, k1 = v_func(state)
        state_k2 = jax.tree_util.tree_map(lambda x, y: x + y * (dt / 2.0), state, k1)
        _, k2 = v_func(state_k2)
        state_k3 = jax.tree_util.tree_map(lambda x, y: x + y * (dt / 2.0), state, k2)
        _, k3 = v_func(state_k3)
        state_k4 = jax.tree_util.tree_map(lambda x, y: x + y * dt, state, k3)
        _, k4 = v_func(state_k4)
        k_sum = jax.tree_util.tree_map(
            lambda k1_val, k2_val, k3_val, k4_val: k1_val + 2.0 * k2_val + 2.0 * k3_val + k4_val,
            k1, k2, k3, k4
        )
        new_state = jax.tree_util.tree_map(
            lambda x, y: x + y * (dt / 6.0),
            S_a, k_sum
        )

        return new_state

    return runge_kutta_step

def get_scan_integration_function(step_foo, inside_iterations, output_transform_function = None):
    @jax.jit
    def step_fn(i, carry):
        state = step_foo(carry)
        return state

    @jax.jit
    def inner_loop_step(start_state, num_steps):
        state = fori_loop(0, num_steps, step_fn, start_state)
        return state

    if output_transform_function is None:
        @jax.jit
        def scan_step(carry, unused_input):
            """(final_x1, final_x2, ...), (evolution_x1_jnp, evolution_x2_jnp, ...)"""

            state = inner_loop_step(
                start_state=carry, num_steps=inside_iterations
            )
            return state, state
        return scan_step
    else:
        @jax.jit
        def scan_step_transformed(carry, unused_input):
            """(final_x1, final_x2, ...), (evolution_x1_jnp, evolution_x2_jnp, ...)"""

            state = inner_loop_step(
                start_state=carry, num_steps=inside_iterations
            )
            return state, output_transform_function(state)
        return scan_step_transformed

@jax.jit
def to_diff(state):
    d = jax.tree_util.tree_map(jnp.zeros_like, state)
    d['time'] = 1.0
    return state, d

class simulation:
    def __init__(self, initials:dict, pipeline_fun, inside_iterations, output_transform_function = None):
        self.state = initials
        self.pipeline = pipeline_fun
        self.inside_iterations =inside_iterations
        self.output_transform_function = output_transform_function
        self.get_integraton_process(output_transform_function)

    def get_integraton_process(self, output_transform_function):
        pipline = self.pipeline
        inside_iterations = self.inside_iterations
        self.scan_step = get_scan_integration_function(pipline, inside_iterations, output_transform_function)
        return self.scan_step
    
    def run(self, iterations):
        iters = jnp.arange(0, iterations)
        self.state, self.history = scan(
            f=self.scan_step,
            init=self.state,
            xs=iters
        )
        return jax.block_until_ready(self.history)