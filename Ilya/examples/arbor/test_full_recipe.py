import os

def change_to_root_directory(max_iterations=100):
    iterations = 0
    while iterations < max_iterations:
        if "root" in os.listdir():
            print(f"'root' найден в директории: {os.getcwd()}")
            return
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        os.chdir(parent_dir)
        iterations += 1
    print("Достигнуто максимальное количество итераций, файл 'root' не найден.")

change_to_root_directory()

print('imports - 1')
import arbor as arb
import pandas as pd
from scripts.arbor_recipes import optimized_recipe
import numpy as np
print('imports - 2')
# paths
params_path = "Datasets/Generated/Optimized_Neural_Params/Optimized_Neural_Params(80).csv"
connectome_path = "Datasets/Generated/neurons_arb"

print("setting sim params")
# setting sim params
neurite_params = pd.read_csv(params_path)
iclamp_schedule = {
    7055857:{
        'soma':arb.iclamp(30 * arb.units.ms, 100 * arb.units.ms, 100.0 * arb.units.nA)
    }
}

print("creating recipe")
# creating recipe
rp = optimized_recipe(
    connectome_dir=connectome_path,
      neurite_params = neurite_params,
        iclamp_schedule=iclamp_schedule)

# Дальше голый арбор
ctx = arb.context()
sim = arb.simulation(rp, ctx)
soma_handles = [sim.sample(gid, 'soma', arb.regular_schedule(0.01*arb.units.ms)) for gid in range(rp.num_cells())]


# запускаем симуляцию
print("starting simulation")
sim.run(tfinal = 100 * arb.units.ms, dt = 0.01 * arb.units.ms)
results = dict()
for h in soma_handles:
    results[h] = sim.samples(h)

print("saving results")
# saving results
np.savez(file = 'arbor_results.npz', **results)