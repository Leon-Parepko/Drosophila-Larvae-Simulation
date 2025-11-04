import os
import pymaid
import networkx as nx
from itertools import product
import pandas as pd
import csv
import json
import numpy as np
from typing import Dict, List, Any, Tuple


class simplified_structure:
    def __init__(self, neuron:pymaid.CatmaidNeuronList, name = None, drop_nodes_inside = False):
        self.neuron:pymaid.CatmaidNeuronList = neuron
        self.nodes:nx.MultiDiGraph = None
        self.name = name
        self.build_structure()
        self.add_connectors_to_graph()
        self.simplify_directed_graph()
        if drop_nodes_inside:
            self.clear_nodes_inside()
        if name is not None:
            self.add_name_to_nodes()
        self.check()

    def clear_nodes_inside(self):
        # Удаляет свойство 'nodes_inside' из всех рёбер графа
        for u, v, k in self.nodes.edges(keys=True):
            if 'nodes_inside' in self.nodes[u][v][k]:
                del self.nodes[u][v][k]['nodes_inside']

    def add_name_to_nodes(self):
        for n in self.nodes.nodes(True, ):
            if n[1]['type'] != 'connector':
                n[1]['name'] = f"{self.name}"

    def check(self):
        # есть root
        if not any(n[1]['type'] == 'root' for n in self.nodes.nodes(True)):
            raise Exception(f"у {self.name} нет root")

    def build_structure(self):
        nodes = self.neuron.nodes
        graph = nx.MultiDiGraph()
        for idt, parent, ntype in zip(nodes['node_id'], nodes['parent_id'], nodes['type']):
            # сома это root
            a = idt >= 0
            b = parent >= 0
            if a:
                graph.add_node(idt, type = ntype) 
            if b:
                graph.add_node(parent, type = ntype)
            if a and b:
                graph.add_edge(idt, parent, nodes_inside = [])
        root_cands = self.neuron.root
        if len(root_cands) != 1:
            raise Exception(f"у {self.name} рут нод {len(root_cands)}")
        self.nodes = graph
        self.nodes.nodes[int(root_cands[0])]['type'] = 'root'
    
    def add_connectors_to_graph(self):
        post_connectors = pymaid.get_connectors(self.neuron, 'postsynaptic_to')
        for idt in post_connectors['connector_id']:
            q = self.neuron.connectors[self.neuron.connectors['connector_id'] == idt]
            self.nodes.add_node(idt, type = 'connector')
            for node_id in q['node_id']:
                self.nodes.add_edge(idt, node_id, nodes_inside = [])

        pre_connectors = pymaid.get_connectors(self.neuron, 'presynaptic_to')
        for idt in pre_connectors['connector_id']:
            q = self.neuron.connectors[self.neuron.connectors['connector_id'] == idt]
            self.nodes.add_node(idt, type = 'connector')
            for node_id in q['node_id']:
                self.nodes.add_edge(node_id, idt, nodes_inside = [])

    def simplify_directed_graph(self):
        def keep_nodes(graph, vid):
            return graph.nodes(True)[vid]['type'] in ('root', 'connector')
        G = self.nodes
        original = len(self.nodes)
        while True:
            # Находим все вершины с in-degree=1 и out-degree=1
            nodes_to_remove = [
                node for node in G.nodes() 
                if G.in_degree(node) == 1 and G.out_degree(node) == 1 and not keep_nodes(G, node)
            ]

            if not nodes_to_remove:
                break # Если таких вершин нет, завершаем

            for node in nodes_to_remove:
                # Если узел уже был удален на предыдущей итерации этого же цикла, пропускаем
                if node not in G: 
                    continue

                # Получаем единственного предшественника и преемника
                # NetworkX гарантирует, что list(predecessors/successors) вернет один элемент,
                # если степень равна 1.
                u = list(G.predecessors(node))
                v = list(G.successors(node))

                if len(u) != 1 or len(v) != 1:
                    raise Exception('Этот эксепшен не должен никогда вызватся, но он вызвался и значит что то пошло не так')
                u = u[0]
                v = v[0]

                nodes_inside = sum((edge[-1]['nodes_inside'] for edge in G.edges(node, data = True)), [])
                    
                if u != v:
                    G.add_edge(u, v, nodes_inside = [node] + nodes_inside)

                G.remove_node(node)

        after = len(self.nodes)
        print('removed', original - after, 'nodes.', f'Efficiency: {round(100*(1 - after/original), 1)}%')

    def save_as_nx_graph(self, path):
        nx.write_gml(self.nodes, path)

class composed_network:
    def __init__(self, paths):
        print(paths)
        self.paths = paths
        self.graphs = {}
        for path in paths:
            print(path)
            self.graphs[path] = nx.read_gml(path)
            for node_id, attr_dict in self.graphs[path].nodes(True):
                filename = os.path.basename(path)
                attr_dict['owner'] = filename

        self.combined_graph = nx.compose_all(self.graphs.values())

    def save_as_gml(self, path):
        nx.write_gml(self.combined_graph, path)

def Pexist(path):
    return os.path.exists(path)

def create_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

class simulation_context:
    def __init__(self, working_dir, neurons, path_to_metadata = None):
        if not Pexist(working_dir):
            create_directory(working_dir)

        self.wd = working_dir
        self.neurons = neurons
        self.input_path = os.path.join(self.wd, 'input')
        self.path_to_neurons = os.path.join(self.input_path, 'neurons')
        self.path_to_full_graph = os.path.join(self.path_to_neurons, 'full', 'full.gml')
        self.path_to_synaptic_table = os.path.join(self.input_path, 'synaptic_table.csv')
        if path_to_metadata is None:
            self.path_to_nodes_metadata = os.path.join(self.input_path, 'nodes_metadata.csv')
        else:
            self.path_to_nodes_metadata = path_to_metadata
        self.output_path = os.path.join(self.wd, 'output')
        self.__catmaid_instance = None
        self.__node_metadata = None

        self.setup()

    def check_neurons(self):
        if not self.get_neurons():
            self.build_full_graph(just_build = True)
    
    @property
    def node_metadata(self):
        if self.__node_metadata is None:
            if not Pexist(self.path_to_nodes_metadata):
                self.get_metadata()
            self.__node_metadata = pd.read_csv(self.path_to_nodes_metadata)
            self.__node_metadata.fillna(0.1)
        return self.__node_metadata

    def setup(self):
        create_directory(self.wd)
        create_directory(self.input_path)
        create_directory(self.path_to_neurons)
        create_directory(os.path.join(self.path_to_neurons, 'full'))
    
    def get_node_neuron(self, node_id):
        return self.node_metadata[self.node_metadata['node_id'] == node_id]

    def process_graph_to_csv(self):
        #TODO переделать что бы не медленный был
        # Load the combined graph
        full_g = nx.read_gml(self.path_to_full_graph)

        # Prepare the CSV file with headers
        with open(self.path_to_synaptic_table, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["pre_neuron_id", "pre_node_id", "connector_id", "post_neuron_id", "post_node_id"])
            writer.writeheader()

            # Process nodes in the combined graph
            for n in full_g.nodes(data=True):
                node_id = int(n[0])
                node_type = n[1].get('type', None)
                # Check if the node is a connector
                if node_type == 'connector':
                    connector_id = node_id
                    from_nodes = [*full_g.predecessors(n[0])]
                    to_nodes = [*full_g.successors(n[0])]

                    # Use itertools.product to generate all pairs of from_nodes and to_nodes
                    for pre_node, post_node in product(from_nodes, to_nodes):
                        pre_node_id = int(pre_node)
                        post_node_id = int(post_node)

                        pre_neuron_row = self.get_node_neuron(pre_node_id)
                        post_neuron_row = self.get_node_neuron(post_node_id)

                        pre_neuron_id = pre_neuron_row['neuron_id'].iloc[0] if not pre_neuron_row.empty else None
                        post_neuron_id = post_neuron_row['neuron_id'].iloc[0] if not post_neuron_row.empty else None

                        writer.writerow({
                            "pre_neuron_id": pre_neuron_id,
                            "pre_node_id": pre_node,
                            "connector_id": connector_id,
                            "post_neuron_id": post_neuron_id,
                            "post_node_id": post_node
                        })

                    # Handle cases where there are only presynaptic or postsynaptic nodes
                    if not to_nodes or not from_nodes:
                        print(f'node {n[0]} has no precessors or no successors -> not added to full graph')

        print(f"Synapse data saved to {self.path_to_synaptic_table}")


    def rm(self, forced_reconnect = False) -> pymaid.CatmaidInstance:
        if self.__catmaid_instance is None or forced_reconnect:

            catmaid_url = 'https://l1em.catmaid.virtualflybrain.org'
            http_user = None
            http_password = None
            project_id = 1

            self.__catmaid_instance = pymaid.CatmaidInstance(catmaid_url, http_user, http_password, project_id)
            print('connected to catmaid')
        return self.__catmaid_instance
    
    def close_catmaid_connection(self):
        #if self.__catmaid_instance is None:
        #    return True
        #self.__catmaid_instance = None
        print('catmaid connection closing is not not implemented')

    def get_neurons(self, forced_rewriting = False, drop_nodes_inside = True):
        rm = None
        q = True
        for neuron_id in self.neurons:
            path_to_neuron = self.path_to_gml(neuron_id)
            if Pexist(path_to_neuron) and not forced_rewriting:
                continue
            q = False
            if rm is None:
                rm = self.rm()
            neuron = pymaid.get_neuron(neuron_id, remote_instance = rm)

            S = simplified_structure(neuron, str(neuron_id), drop_nodes_inside)
            S.save_as_nx_graph(path_to_neuron)
        
        self.close_catmaid_connection()
        return q

    def gml_path(self, neuron_id):
        p = os.path.join(self.path_to_neurons, f'{neuron_id}.gml')
        if not Pexist(p):
            print(f"cannot find {p} -> getting all neurons")
            self.get_neurons(self)
            self.build_full_graph(forced = True)
        return p

    def build_full_graph(self, forced = False, just_build = False):
        if Pexist(self.path_to_full_graph):
            return
        if not just_build:
            self.get_neurons()
        cn = composed_network([self.gml_path(n) for n in self.neurons])
        cn.save_as_gml(self.path_to_full_graph)

    @property
    def synaptic_table(self):
        if not Pexist(self.path_to_synaptic_table):
            self.build_synaptic_table()
        return pd.read_csv(self.path_to_synaptic_table)

    def build_synaptic_table(self):
        self.get_metadata()
        self.build_full_graph()
        self.process_graph_to_csv()

    def get_arbor_recipe(self):
        self.build_full_graph()

    def path_to_gml(self, ind):
        return os.path.join(self.path_to_neurons, str(ind) + '.gml')

    def get_metadata(self):
        rm = self.rm()

        # neurons = pymaid.find_neurons(remote_instance = rm)
        all_skids = self.neurons

        # Containers for metadata
        #neurons_metadata = []
        nodes_metadata = []

        for skid in all_skids:
            try:
                neuron = pymaid.get_neuron(skid, remote_instance = rm)
                
                # --- Neuron-level metadata ---
                # neuron_metadata = {
                #     "neuron_id": neuron.id,
                #     "name": neuron.name,
                #     "type": neuron.type,
                #     "n_nodes": neuron.n_nodes,
                #     "n_connectors": neuron.n_connectors,
                #     "n_branches": neuron.n_branches,
                #     "n_leafs": neuron.n_leafs,
                #     "cable_length": neuron.cable_length,
                #     "annotation": neuron.annotations
                # }
                # neurons_metadata.append(neuron_metadata)

                # --- Node-level metadata ---
                nodes = neuron.nodes[["node_id", "x", "y", "z", "radius", "type"]].copy()
                nodes["neuron_id"] = neuron.id  # link to parent neuron

                # Fix radius: None if NaN or negative
                nodes["radius"] = nodes["radius"].apply(
                    lambda r: None if pd.isna(r) or r <= 0 else r
                )

                # --- Connector metadata ---
                connectors = pymaid.get_connectors(neuron, remote_instance = rm)[["connector_id", "x", "y", "z", "type"]].copy()
                connectors = connectors.rename(columns={"connector_id": "node_id"})
                connectors["radius"] = None  # no radius for connectors
                connectors["neuron_id"] = neuron.id  # link to parent neuron

                # Merge nodes + connectors into one table
                node_info = pd.concat([nodes, connectors], ignore_index=True)

                nodes_metadata.append(node_info)

            except Exception as e:
                print(f"FAIL {skid}: {e}")

        # Convert lists to DataFrames
        #neurons_metadata = pd.DataFrame(neurons_metadata)
        nodes_metadata = pd.concat(nodes_metadata, ignore_index=True)
        nodes_metadata.to_csv(self.path_to_nodes_metadata)


class simulation_context_jax(simulation_context):
    def __init__(self, working_dir, neurons, path_to_metadata=None):
        
        super().__init__(working_dir, neurons, path_to_metadata)
        self.jax_path = create_directory(os.path.join(self.wd, 'jax'))
        self.path_to_reindexed_graph = os.path.join(self.jax_path, 'reindexed_graph.gml')
        self.path_to_mapping = os.path.join(self.jax_path, 'node_mapping.json')
        
        self.path_H_to_H_edges = os.path.join(self.jax_path, 'H_to_H_edges.npy')
        self.path_H_to_syn_edges = os.path.join(self.jax_path, 'H_to_syn_edges.npy')
        self.path_syn_to_H_edges = os.path.join(self.jax_path, 'syn_to_H_edges.npy')
        self.path_num_nodes = os.path.join(self.jax_path, 'num_nodes.json')


    def reindex_graph(self, force_rewrite: bool = False) -> nx.MultiDiGraph:
        self.build_full_graph()
        
        if not force_rewrite and Pexist(self.path_to_reindexed_graph):
            return nx.read_gml(self.path_to_reindexed_graph)

        full_g = nx.read_gml(self.path_to_full_graph)
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(full_g.nodes())}
        new_to_old = {v: k for k, v in old_to_new.items()}

        reindexed_graph = nx.relabel_nodes(full_g, old_to_new, copy=True)
        
        with open(self.path_to_mapping, 'w', encoding='utf-8') as json_file:
            old_to_new_str = {str(k): v for k, v in old_to_new.items()}
            new_to_old_str = {str(k): v for k, v in new_to_old.items()}
            json.dump({"old_to_new": old_to_new_str, "new_to_old": new_to_old_str}, json_file, indent=4)
        print(f"Node mapping saved to {self.path_to_mapping}")

        nx.write_gml(reindexed_graph, self.path_to_reindexed_graph)
        print(f"Reindexed graph saved to {self.path_to_reindexed_graph}")

        return reindexed_graph

    def prepare_jax_arrays(self, force_rewrite: bool = False):
        if not force_rewrite and Pexist(self.path_H_to_H_edges):
            print("JAX arrays already exist. Use force_rewrite=True to regenerate.")
            return

        reindexed_graph = self.reindex_graph(force_rewrite=force_rewrite)
        
        hh_nodes: List[int] = []
        syn_nodes: List[int] = []

        # new_global_id -> ('H' | 'S')
        node_type_map: Dict[int, str] = {}
        
        for node_id, data in reindexed_graph.nodes(data=True):
            node_type = data.get('type')
            
            if node_type == 'connector':
                syn_nodes.append(node_id)
                node_type_map[node_id] = 'S'
            else:
                hh_nodes.append(node_id)
                node_type_map[node_id] = 'H'

        hh_map = {node_id: i for i, node_id in enumerate(hh_nodes)}
        syn_map = {node_id: i for i, node_id in enumerate(syn_nodes)}
        
        num_H = len(hh_nodes)
        num_syn = len(syn_nodes)
        
        
        H_to_H_edges: List[Tuple[int, int]] = []
        H_to_syn_edges: List[Tuple[int, int]] = []
        syn_to_H_edges: List[Tuple[int, int]] = []

        for u, v in reindexed_graph.edges():
            u_type = node_type_map.get(u)
            v_type = node_type_map.get(v)

            if u_type == 'H' and v_type == 'H':
                # H -> H
                u_local = hh_map[u]
                v_local = hh_map[v]
                #TODO optimize memory usage
                if (v_local, u_local) not in H_to_H_edges:
                     H_to_H_edges.append((u_local, v_local))
                
            elif u_type == 'H' and v_type == 'S':
                # H -> Syn
                u_local = hh_map[u]
                v_local = syn_map[v]
                H_to_syn_edges.append((u_local, v_local))

            elif u_type == 'S' and v_type == 'H':
                # Syn -> H
                u_local = syn_map[u]
                v_local = hh_map[v]
                syn_to_H_edges.append((u_local, v_local))
        np.save(self.path_H_to_H_edges, np.array(H_to_H_edges, dtype=np.int32))
        np.save(self.path_H_to_syn_edges, np.array(H_to_syn_edges, dtype=np.int32))
        np.save(self.path_syn_to_H_edges, np.array(syn_to_H_edges, dtype=np.int32))
        
        print(f"H_to_H_edges saved: {len(H_to_H_edges)} edges (undirected representation)")
        print(f"H_to_syn_edges saved: {len(H_to_syn_edges)} edges")
        print(f"Syn_to_H_edges saved: {len(syn_to_H_edges)} edges")

        # Сохранение метаданных о размерах
        with open(self.path_num_nodes, 'w', encoding='utf-8') as f:
            json.dump({"num_H": num_H, "num_syn": num_syn}, f)
        
        print(f"Node counts saved: num_H={num_H}, num_syn={num_syn}")

    def get_jax_context(self) -> Dict[str, Any]:
        if not Pexist(self.path_H_to_H_edges):
            self.prepare_jax_arrays()
        

        H_to_H = np.load(self.path_H_to_H_edges)
        H_to_syn = np.load(self.path_H_to_syn_edges)
        syn_to_H = np.load(self.path_syn_to_H_edges)
        

        with open(self.path_num_nodes, 'r', encoding='utf-8') as f:
            num_nodes = json.load(f)
            num_H = num_nodes["num_H"]
            num_syn = num_nodes["num_syn"]


        with open(self.path_to_mapping, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            
        return {
            "num_H": num_H,
            "num_syn": num_syn,
            "H_to_H": H_to_H,
            "V_to_syn": H_to_syn,
            "syn_to_V": syn_to_H,
            "mapping": mapping,
        }