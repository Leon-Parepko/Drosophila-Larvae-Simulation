import numpy as np
import pandas as pd

def generate_connections(groups, group_complexity, group_density = 0.5, noise = 1, noise_density = 0.3, group_recurent_rate = 0.5):
    return \
    np.repeat(np.repeat((1-group_recurent_rate)*(np.random.sample([groups, groups]) < group_density) + group_recurent_rate*np.eye(groups, groups), group_complexity, 0), group_complexity, 1) * np.random.uniform(-1, 1, [group_complexity*groups]*2) + \
    noise * np.random.uniform(-1, 1, [group_complexity*groups]*2) * (np.random.sample([group_complexity*groups]*2) < noise_density) # skip connections

class activity_system:
    def __init__(self, group_count, group_complexity, group_dens=0.5, noise_density=0.3, group_self_connections = 0.1, activity_density = 0.5, noise_level = 1.0, dt = 0.01):
        total = group_count*group_complexity
        self.x = (np.random.sample(total) < activity_density).astype(np.float32)
        self.x /= np.sum(self.x)
        self.w = generate_connections(group_count, group_complexity, group_dens, noise_density=noise_density, group_recurent_rate=group_self_connections, noise=noise_level)
        self.dt = dt
    def update(self):
        self.x += self.dt*(self.w - self.w.T) @ self.x
        #self.x += 0.01*(1 - np.sum(self.x)) * (self.x/(np.sum(self.x**2)**0.5 + 1))
        self.x /= np.sum(self.x**2)**0.5
        return self.x

def some_activity_system(group_count = 100, group_complexity = 20, dt = 0.01):
    return activity_system(group_count, group_complexity, noise_density=0.1, noise_level=1.0, group_self_connections=0.5, dt = dt)

def generate_activity_df(system:activity_system, size:int, df = None) -> pd.DataFrame:
    if df is None:
        df = pd.DataFrame(columns=[i for i in range(system.x.shape[0])])
    for i in range(size):
        system.update()
        df.loc[i] = np.copy(system.x)
    return df