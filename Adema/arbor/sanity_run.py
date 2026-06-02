import os
import sys

def change_to_root_directory(max_iterations=100):
    iterations = 0
    while iterations < max_iterations:
        if "root" in os.listdir():
            print(f"'root' найден в директории: {os.getcwd()}")
            return
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        if parent_dir == os.getcwd():
            break
        os.chdir(parent_dir)
        iterations += 1
    print("root not found")

change_to_root_directory()

sys.path.insert(0, os.path.join(os.getcwd(), "Adema", "arbor"))

print("imports - 1")
import arbor as arb
import numpy as np
import matplotlib.pyplot as plt
from arbor_recipes import optimized_recipe
print("imports - 2")

connectome_path = "Datasets/Generated/neurons_arb"
output_dir = "arbor_outputs_no_params_small_dt"
os.makedirs(output_dir, exist_ok=True)

stimulated_neuron_id = 7055857

iclamp_schedule = {
    stimulated_neuron_id: {
        "soma": arb.iclamp(30 * arb.units.ms, 20 * arb.units.ms, 5.0 * arb.units.nA)
    }
}

print("creating recipe WITHOUT neurite params")
rp = optimized_recipe(
    connectome_dir=connectome_path,
    neurite_params=None,
    iclamp_schedule=iclamp_schedule,
    cv_max_extent=5.0,
)

num_cells = rp.num_cells()
print("num_cells:", num_cells)

gid_matches = [gid for gid, nid in rp.gid_to_neuron_id.items() if nid == stimulated_neuron_id]
print("stimulated neuron_id:", stimulated_neuron_id)
print("matching Arbor gids:", gid_matches)

ctx = arb.context()
sim = arb.simulation(rp, ctx)

print("registering samplers")
sample_dt = 0.01 * arb.units.ms
soma_handles = [
    sim.sample(gid, "soma", arb.regular_schedule(sample_dt))
    for gid in range(num_cells)
]

print("starting simulation")
sim.run(tfinal=60 * arb.units.ms, dt=0.001 * arb.units.ms)
print("simulation finished")

all_values = []
all_gids = []
time = None

print("collecting samples")
for gid, h in enumerate(soma_handles):
    samples = sim.samples(h)
    if not samples:
        continue

    data, _ = samples[0]

    if time is None:
        time = data[:, 0]

    all_values.append(data[:, 1])
    all_gids.append(gid)

values_matrix = np.vstack(all_values)
gids = np.array(all_gids)

print("values_matrix shape:", values_matrix.shape)
print("total NaN values:", np.isnan(values_matrix).sum())
print("total Inf values:", np.isinf(values_matrix).sum())

valid_mask = np.isfinite(values_matrix).all(axis=1)
values_valid = values_matrix[valid_mask]
gids_valid = gids[valid_mask]

print("valid neurons:", len(gids_valid))
print("invalid neurons:", len(gids) - len(gids_valid))

if len(gids_valid) == 0:
    raise RuntimeError("No valid neuron traces found.")

print("global min:", np.min(values_valid))
print("global max:", np.max(values_valid))

if len(gid_matches) == 0:
    raise RuntimeError("Stimulated neuron not found")

stim_gid = gid_matches[0]

if stim_gid not in gids_valid:
    raise RuntimeError(f"Stimulated neuron gid {stim_gid} is invalid")

stim_idx = np.where(gids_valid == stim_gid)[0][0]
stim_trace = values_valid[stim_idx]

baseline = stim_trace[time < 30].mean()
peak = stim_trace[(time >= 30) & (time <= 50)].max()
amp = peak - baseline

print("stim_gid:", stim_gid)
print("baseline:", baseline)
print("peak:", peak)
print("amplitude:", amp)

plt.figure(figsize=(12, 4))
plt.plot(time, stim_trace)
plt.axvspan(30, 50, alpha=0.2)
plt.title(f"Stimulated neuron (gid={stim_gid}), amp={amp:.6f} mV")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "stimulated_neuron.png"), dpi=300)
plt.show()

max_per_neuron = np.max(values_valid, axis=1)
top_idx = np.argsort(max_per_neuron)[-10:][::-1]

plt.figure(figsize=(12, 6))
for idx in top_idx:
    plt.plot(time, values_valid[idx])
plt.title("Top 10 responding neurons")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top10.png"), dpi=300)
plt.show()

mean_trace = values_valid.mean(axis=0)

plt.figure(figsize=(12, 4))
plt.plot(time, mean_trace)
plt.title("Population mean voltage")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "population_mean.png"), dpi=300)
plt.show()

print("done")