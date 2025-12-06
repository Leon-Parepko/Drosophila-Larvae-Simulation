import scripts.nj.graph_to_arrays as ga
import networkx as nx
import numpy as np
import pandas as pd

def process_basic(path_to_full, path_to_save, path_to_metadata):
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


    metadata = pd.read_csv(path_to_metadata)
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
    

def process_params2025d05(path_to_full, path_to_save, path_to_metadata):
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


    metadata = pd.read_csv(path_to_metadata)
    global_mapping = res['mapping']
    metadata = metadata.fillna(10.0) # 10.0 as basic radius
    metadata['new_index'] = metadata.apply(lambda row:global_mapping['cable'].get(str(row['node_id'])), axis = 1)
    metadata = metadata.dropna(subset=['new_index']).sort_values('new_index')
    #metadata = metadata.set_index('new_index').sort_index()


    all_somas = metadata[metadata['type'] == 'root']['node_id'].to_numpy()
    stom = [(int(soma), int(global_mapping['cable'][str(soma)])) for soma in all_somas]
    stom = np.array(stom)

    gnabar_hhtom = [(int(soma), int(global_mapping['cable'][str(soma)])) for soma in all_somas]

    ga.save_jax_arrays(res, path_to_save, {"stom":stom, # сома_global_id, сома_cabble_id
                                            'gnabar_hh':metadata['gnabar_hh'].to_numpy(),
                                            'gkbar_hh':metadata['gkbar_hh'].to_numpy(),
                                            'gl_hh':metadata['gl_hh'].to_numpy(),
                                            'L':metadata['L'].to_numpy(),
                                            'Ra':metadata['Ra'].to_numpy(),
                                            'diam':metadata['diam'].to_numpy(),
                                            'el_hh':metadata['el_hh'].to_numpy(),
                                            })