# Drosophila Larvae Simulation

A research project to build a **full digital model of an organism** — the Drosophila larva (L1 EM, first instar larva electron microscopy connectome) — from connectome, morphologies, and synaptic data. The project aims to create a bottom-up in silico implementation of the whole organism to understand living system mechanisms and run computational experiments.

---

## Main idea

**The goal is to obtain a digital copy of a living organism** (Drosophila larva) in the computer: a single model in which the neural network, built from electron microscopy and neuroanatomical databases (e.g. CATMAID), reproduces and predicts activity and, where needed, behaviour. Inputs are the EM connectome and neuron morphologies. Modelling is done at several levels of detail: from point neurons (integrate-and-fire) to compartmental models with Hodgkin–Huxley channels on full morphologies, so as to link structure and function and eventually close the loop with sensory input and motor output.

---

## Goals and tasks

- **Unified data pipeline**: preparation of neuron graphs (GML), connectivity matrices, metadata, synaptic tables, and other CATMAID-derived data from raw datasets.
- **Point modelling**: simulations of the network as a graph of point neurons (LIF, etc.) in NEST for fast screening of topology, excitation/inhibition, and activity metrics.
- **Compartmental modelling**: simulations on morphologies with Hodgkin–Huxley channels in Arbor and NEURON for biologically detailed scenarios.
- **Visualisation and analysis**: visualisation, saving of results (NPZ/PKL), and statistical analysis of activity.
- **Integration into a whole-organism model**: progress towards a full digital copy of the organism with closed sensorimotor loops.

---

## Why this is needed

- **Digital twin of the organism**: ability to reproduce and predict neural activity and eventually behaviour in silico, and to run experiments without intervening in the living animal (…TODO…).
- **Structure–function relationship**: testing how far the known connectome and neuron types determine activity patterns under different stimuli and parameters.
- **Comparing modelling levels**: understanding when point models (NEST) suffice and when compartmental models (Arbor/NEURON) are needed.
- **Reproducibility**: a single repository with data, preparation scripts, and simulation recipes to reproduce experiments (…TODO…).

---

## Repository structure

```
Drosophila-Larvae-Simulation/
│
├── Datasets/
│   ├── Original/                           # Raw data
│   │   ├── neurons/                        # GML morphologies (one file per neuron)
│   │   ├── neurons.npz                     # Same morphologies in a single file
│   │   ├── aa_connectivity_matrix.csv      # axon–axon connectors
│   │   ├── ad_connectivity_matrix.csv      # axon–dendrite connectors
│   │   ├── da_connectivity_matrix.csv      # dendrite–axon connectors
│   │   ├── dd_connectivity_matrix.csv      # dendrite–dendrite connectors
│   │   ├── all-all_connectivity_matrix.csv # all connectors
│   │   ├── Metadata(auto).pkl              # Neuron metadata (auto-built from CATMAID)
│   │   ├── s2.csv
│   │   ├── s3.csv
│   │   ├── s4.csv
│   │   ├── inputs.csv
│   │   └── outputs.csv
│   └── Generated/
│       ├── Metadata_Neurons(manual).csv     # Neuron-level metadata
│       ├── Metadata_Nodes(manual).csv       # Synaptic-level (node) metadata
│       ├── Syn_Conns(manual).csv            # Synaptic connections
│       ├── Syn_Types(By_Cell_Types).csv     # Synapse types by cell type (exc./inh.)
│       ├── aa.pkl, ad.pkl, da.pkl, dd.pkl   # Serialised connectivity graphs/matrices
│       ├── complete_graph(3k).gml           # Full synaptic-level graph of the connectome
│       ├── complete_graph(5k).gml           # Full synaptic-level graph (extended)
│       ├── test_graph_2n.gml … test_graph_20n.gml  # Test graphs
│       ├── Optimized_Neural_Params/         # Fitted parameters (10, 30, 50, … % connectivity)
│       └── Experiments/                     # Simulation outputs: .npz and .pkl (soma voltages)
│
├── scripts/
│   ├── data_preparation.py       # TODO
│   ├── arbor_recipes.py          # Arbor simulation module
│   ├── NEURON_Sim_Wrapper.py     # NEURON simulation module
│   ├── Sim_Visializations.py     # 2D/3D activity visualisation
│   └── to_obj.py                 # Export graph to OBJ (x,y,z metadata)
│
├── Visualizations/
│
├── Ilya/
│   └── ...TODO...      # TODO Will be finished at the end
│
├── Leon/
│   └── ...TODO...      # TODO Will be finished at the end
│
├── Rail/
│   └── ...TODO...      # TODO Will be finished at the end
│
└── Adema/
    └── ...TODO...      # TODO Will be finished at the end
```

---

## Data we work with

We represent the connectome and morphologies at two graph levels and attach **metadata** to both. Primary connectome data: …TODO…

### 1. Neuron-level graph

- **Nodes** = neurons (identified by neuron ID / skid).
- **Edges** = connectivity between neurons; edge weights are synapse counts (or derived quantities) between each pair.
- **Sources**: connectivity matrices `aa`, `ad`, `da`, `dd` (axon–axon, axon–dendrite, dendrite–axon, dendrite–dendrite) and `all-all_connectivity_matrix.csv`. From these we build a single directed graph used for **point modelling** (e.g. in NEST): one node per neuron, edges weighted by connectivity.
- **Statistics**: ~3k neurons, hundreds of thousands of synapses; a–d dominates (~66%), then a–a and d–d (~26%), d–a rare (~2%); most connections are weak (1–2 synapses).

### 2. Synaptic-level graph

- **Nodes** = morphological compartments and connectors: cable nodes (root, branch, slab, end) and connector nodes (synaptic sites). Each neuron is a tree of cable nodes; connectors link pre- and post-synaptic nodes across neurons.
- **Edges** = parent–child in the cable tree, and connector–(pre/post) node links. Full geometry (x, y, z, radius) is stored per node.
- **Sources**: GML files in `neurons/` (one file per neuron) and combined graphs such as `complete_graph(3k).gml`; synaptic tables e.g. `Syn_Conns(manual).csv` with columns pre_neuron_id, pre_node_id, connector_id, post_neuron_id, post_node_id. This level is used for **compartmental modelling** (Arbor, NEURON): placement of ion channels and synapses on specific compartments.
- **Formats**: GML for graphs; CSV for synaptic tables; NPZ for array-based exports used in simulations.

### 3. Metadata

- **Neuron-level**: cell type (KCs, PNs, LNs, MBINs, sensory, ascending, etc.), excitation/inhibition (exc/inh), input/output flags, axon/dendrite ratios. Sources: `s2.csv`, `s3.csv`, `s4.csv`, `inputs.csv`, `outputs.csv`, `Metadata(auto).pkl`, `Metadata_Neurons(manual).csv`. Used to annotate the neuron-level graph (e.g. is_inh, is_input, is_output, signal_depth) and to assign synapse types.
- **Synaptic-level (node)**: coordinates and radii (x, y, z, radius) for each node, used to build compartment trees. Sources: `Metadata_Nodes(manual).csv` and metadata derived from CATMAID. `Syn_Types(By_Cell_Types).csv` gives synapse type by cell type (exc./inh.) for modelling.
- **Formats**: CSV, PKL (e.g. Metadata(auto).pkl), and fields inside GML/Generated files.

---

## Point modelling (NEST)

Point modelling is implemented in **Adema** using **NEST**.

- **Input**: the **neuron-level graph** (built from aa, ad, da, dd matrices) plus **metadata** (cell types, input/output, exc/inh from s2–s4, outputs).
- **Idea**: neurons are point objects (LIF or similar); connectivity is the graph above. Edge weights = synapse counts; delays and exc/inh are set from metadata.
- **Main files**: **nest.ipynb** (load matrices, build graph, annotate is_inh, is_input, is_output, signal_depth, signal_direction; create NEST network, run simulations), **metrics.ipynb** (connectome statistics), **neuron_test_graph.py** (test on a small set of neurons).
- **Parameters**: neuron_params (C_m, tau_m, t_ref, E_L, V_reset, V_th), synaptic weights and delays; CELLTYPE_MAP for exc/inh.
- **Goal**: fast screening of topology and excitation/inhibition, propagation from sensory inputs, activity metrics (e.g. echo duration) without morphology.

---

## Compartmental modelling (Arbor and NEURON)

Compartmental modelling uses the **synaptic-level graph** (GML morphologies, node geometry) and **metadata** (node coordinates/radii, synaptic tables). Neurites are trees of segments; Hodgkin–Huxley channels and synapses are placed on compartments.

### Arbor

- **Location**: **scripts/arbor_recipes.py**, and variants in **Ilya/arbor_recipes/**.
- **Input**: **synaptic-level** description (GML per cell, node metadata for geometry) and synaptic table (pre/post neuron_id, node_id). Class `basic_recipe` implements `arbor.recipe`: for each GID it loads GML, builds segment_tree from nodes (root → branch/end), reads coordinates and radius from node metadata; places synapses from the table; optionally adds iclamp and threshold detectors. Cables are cable_cell with mechanisms (hh, expsyn, etc.). Data come from simulation_context in data_preparation (full.gml, synaptic_table.csv, nodes_metadata).

### NEURON

- **Location**: **scripts/NEURON_Sim_Wrapper.py**; called from notebooks in **Leon/** (e.g. Experiments, NEURON_With_Wrapper*).
- **Input**: **synaptic-level** morphologies (GML in Datasets/Original/neurons/) and **Metadata(auto).pkl** (connector–node mapping). Class `Network` takes a list of neuron_ids, path to GML directory, and path to Metadata(auto).pkl. For each neuron it builds the section tree from GML; nseg is set by the λ-rule. HH mechanisms and synapses are placed on segments according to connector metadata; voltage traces and stimuli are supported. Visualisation: **Sim_Visializations.py** or similar.

Both approaches study the effect of morphology and synapse placement on dynamics and can be compared with point models (Adema/NEST).

---

## Requirements and running

- **Python**: 3.9
- **Dependencies**: see `Ilya/requirements.txt`, `Leon/requirements.txt`; for NEST install NEST; for NEURON install NEURON with Python interface; for Arbor install arbor.
- **Additional**: pymaid (for CATMAID), networkx, pandas, numpy, matplotlib, plotly.

For many notebooks the working directory is the repository root; data paths are given relative to it.

---

## License

The project is distributed under the MIT License (see the LICENSE file).

---

## Citation

When using the Drosophila larva connectome data, please cite the primary dataset publications (…TODO…). When using the code and results of this research project, please cite: …TODO…
