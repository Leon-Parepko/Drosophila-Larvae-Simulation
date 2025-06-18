import numpy as np
import pandas as pd

def generate_activity_df(acivity, df=None) -> tuple[pd.DataFrame]:  # df_w, #df
    if df is None:
        df = pd.DataFrame(columns=[i for i in range(acivity.shape[1])])
    for i in range(acivity.shape[0]):
        df.loc[i] = acivity[i]

    return df


def generate_connections(
    groups,
    group_complexity,
    group_density=0.5,
    noise=1,
    noise_density=0.3,
    group_recurent_rate=0.5,
):
    return np.repeat(
        np.repeat(
            (1 - group_recurent_rate)
            * (np.random.sample([groups, groups]) < group_density)
            + group_recurent_rate * np.eye(groups, groups),
            group_complexity,
            0,
        ),
        group_complexity,
        1,
    ) * np.random.uniform(
        -1, 1, [group_complexity * groups] * 2
    ) + noise * np.random.uniform(-1, 1, [group_complexity * groups] * 2) * (
        np.random.sample([group_complexity * groups] * 2) < noise_density
    )  # skip connections


def generate_multiple_connections(C):
    N = C.shape[0]
    return (np.random.uniform(-1, 1, [N] * 3)) * C


def generate_activity(W, t_end=10, dt=0.001):
    s = xwx(W, dt=dt)
    s.x = np.random.uniform(-1, 1, W.shape[0])
    h = s.solve(t_end)
    return h


def randomW(N):
    W = np.random.uniform(-1, 1, (N, N, N)) - 100 * np.eye(N)
    W = generate_multiple_connections(W)
    W = W - W.T
    return W


class solver:
    def __init__(self, dt=0.1, **rules):
        self.rules = rules
        self.params = rules.keys()
        self.dt = dt

    def update(self):
        self.set_to(
            **{
                name: getattr(self, name)
                + self.rules[name](**{name: getattr(self, name) for name in self.rules})
                * self.dt
                for name in self.rules
            }
        )

    def set_to(self, **values):
        for j in values:
            setattr(self, j, values[j])

    def solve(self, t_end, t_start=0):
        self.t = t_start
        history = {name: [] for name in self.rules}
        history["t"] = []
        while self.t < t_end:
            self.update()
            for j in history:
                history[j].append(getattr(self, j))
            self.t += self.dt
        delattr(self, "t")
        for j in history:
            history[j] = np.array(history[j])
        return history


class xwx(solver):
    def __init__(self, W, dt=0.1):
        self.W = W
        super().__init__(dt, x=lambda x: x @ self.W @ x)


W = generate_connections(
    100, 4, group_density=1.0, noise=0.1, noise_density=1.0, group_recurent_rate=0.9
)
W = generate_multiple_connections(W)
W = (W - W.T)
dt = 0.001
activity = generate_activity(W, t_end=1.0, dt=dt)
df_a = generate_activity_df(activity["x"])
df_a.to_csv("Ilya/linar_alg/compress/a_big_one.csv")