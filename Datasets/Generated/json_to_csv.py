import json
import pandas as pd

neuron_list = ['7055857', '1805418', '14260575', '5835799', '10160250', '7840203', '5019924', '13986477', '10167078', '7982896', '4119387', '17591442', '4227544', '10495502', '8069478', '3913629', '11279244', '16846805', '8980589', '3664102']



# Get the neurite params for the ones
with open("./Datasets/Generated/Optimized_Neural_Params(95).json", "r") as f:
    data = json.load(f)

rows = []
for neuron_id, content in data["results"].items():
    p = content["best_params"]
    rows.append({
        "neuron_id": int(neuron_id),
        "gnabar_hh": p["dend_gnabar_hh"],
        "gkbar_hh": p["dend_gkbar_hh"],
        "gl_hh": p["dend_gl_hh"]
    })

neurites_params = pd.DataFrame(rows)

# Cast to string
neurites_params["neuron_id"] = neurites_params["neuron_id"].astype(str)

# Get only needed 20 neurons
neurites_params = neurites_params[neurites_params["neuron_id"].isin(neuron_list)]



# Set the rest of params manually
neurites_params['L'] = 5.0
neurites_params['diam'] = 1.0
neurites_params['Ra'] = 100.0
neurites_params['cm'] = 1.0
neurites_params['el_hh'] = -70.0

metadata = pd.read_csv("Datasets/Generated/metadata/nodes_metadata_20n.csv")
neurites_params['neuron_id'] = neurites_params['neuron_id'].astype(int)
total_metadata = pd.merge(metadata, neurites_params, on = 'neuron_id')
neurites_params.to_csv("Datasets/Generated/metadata/neurons_hh_params_2025d5.csv")
total_metadata.to_csv("Datasets/Generated/metadata/nodes_metadata_20n_with_params.csv")