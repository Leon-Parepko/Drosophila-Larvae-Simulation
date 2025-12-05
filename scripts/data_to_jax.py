import scripts.nj.graph_to_arrays as ga
import networkx as nx
import numpy as np
import pandas as pd
path_to_full = "Ilya/trash/del_this_syn/input/neurons/full/full.gml"
path_to_save = "Datasets/Generated/jax/2025d5/20n.npz"

def process_basic(path_to_full, path_to_save):
    '''
    node_type_groups = {
    'cable':['branch', 'root', 'slab', 'end'],
    'alpha':['connector']}

    edge_directedness={'cable': {'cable': False},}


    stom, x, y, z, r


    '''
    graph = nx.read_gml(path_to_full)

    res = ga.process_graph_to_core_arrays(graph, node_type_groups = {
        'cable':['branch', 'root', 'slab', 'end'],
        'alpha':['connector']
    }, edge_directedness={'cable': {'cable': False},})


    metadata = pd.read_csv("Datasets/Generated/trash/input/nodes_metadata.csv")
    global_mapping = res['mapping']
    metadata = metadata.fillna(10.0) # 10.0 as basic radius
    metadata['new_index'] = metadata.apply(lambda row:global_mapping['cable'].get(str(row['node_id'])), axis = 1)
    metadata = metadata.dropna(subset=['new_index'])
    metadata = metadata.set_index('new_index').sort_index()


    all_somas = metadata[metadata['type'] == 'root']['node_id'].to_numpy()
    stom = [(int(soma), int(global_mapping['cable'][str(soma)])) for soma in all_somas]
    stom = np.array(stom)

    ga.save_jax_arrays(res, path_to_save, {"stom":stom, # сома_global_id, сома_cabble_id
                                            'x':metadata['x'].to_numpy(),
                                            'y':metadata['y'].to_numpy(),
                                            'z':metadata['z'].to_numpy(),
                                            'r':metadata['radius'].to_numpy()})