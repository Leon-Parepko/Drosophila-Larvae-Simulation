import arbor as arb
import pandas as pd
import os
import json
import pickle
from multiprocessing import Process
from random import random

def mk_dir(path, name):
    p = os.path.join(path, name)
    os.mkdir(p)
    return p

class converter:
    def __init__(self, graphs, output_path, metadata, syn_data, replace_nan_by = {'radius':1.0}, keys = None, neurite_params = None, cable_type = 'hh', synapce_type = 'expsyn'):
        self.n_to_arb_coords = dict()
        self.graph = graphs
        self.output_path = output_path
        self.node_metadata:pd.DataFrame = metadata
        for k, v in replace_nan_by.items():
            self.node_metadata[k] = self.node_metadata[k].fillna(v)
        self.keys = tuple(graphs.keys()) if keys is None else tuple(keys)
        self.neurite_params = neurite_params
        self.syn_data = syn_data
        self.cablet = cable_type
        self.synt = synapce_type
        self.id_to_gid = {
            k:n for n, k in enumerate(self.keys)
        }
        with open(os.path.join(self.output_path, "gid_mapping.json"), "w") as f:
            json.dump(self.id_to_gid, f)
        

    def __get_node_geometry(self, ind):
        '''x, y, z, r'''
        ind = int(ind)
        s = self.node_metadata[self.node_metadata['node_id'] == ind]
        return s.to_numpy()[0, 1:5]

    def __in_graph(self, parent_segment_id, current_node, node_to_segment, tree, tp):
        for to_ in tp.predecessors(current_node):
            if tp.nodes[to_].get('type') == 'connector':
                continue
            segment_id = tree.append(parent_segment_id, arb.mpoint(*self.__get_node_geometry(current_node)),
                                    arb.mpoint(*self.__get_node_geometry(to_)), tag=int(to_))
            node_to_segment[to_] = segment_id
            self.__in_graph(segment_id, to_, node_to_segment, tree, tp)

    def __mr(self, l):
        for i in l:
            self.run(i)

    def convert(self, num_t):
        procs = []
        for i in range(0, len(self.keys), num_t):
            lk = self.keys[i:i + num_t]
            procs.append(Process(target = self.__mr, args = (lk,)))
        for p in procs:
            p.start()
        for p in procs:
            p.join()

    def decorator(self, ind, node_to_segment):
        decor = arb.decor()
        neuron_id = ind
        # cable properties per neuron 
        if self.neurite_params is not None and neuron_id in self.neurite_params.index:
            params = self.neurite_params.loc[neuron_id]
            # 1) Cable properties: cm, Ra
            # In Arbor, axial resistivity is usually called rL (Ω·cm),
            # so we map your DataFrame's "Ra" -> rL here:
            if "cm" in params:
                decor.set_property(cm=float(params["cm"]))
            if "Ra" in params:
                decor.set_property(rL=float(params["Ra"]))

            # 2) HH mechanism parameters for this cell
            # The exact parameter names depend on the 'hh' mechanism;
            # typical names are gnabar, gkbar, gl, el. Check via:
            #   cat = arb.default_catalogue()
            #   info = cat['hh']  (or arb.mech_info)
            hh_kwargs = {}
            if "gnabar_hh" in params:
                hh_kwargs["gnabar"] = float(params["gnabar_hh"])
            if "gkbar_hh" in params:
                hh_kwargs["gkbar"] = float(params["gkbar_hh"])
            if "gl_hh" in params:
                hh_kwargs["gl"] = float(params["gl_hh"])
            if "el_hh" in params:
                hh_kwargs["el"] = float(params["el_hh"])

            decor.paint("(all)", arb.density(self.cablet, **hh_kwargs))
        else:
            # fallback: same parameters for everyone
            decor.paint("(all)", arb.density(self.cablet))


        neuron_id = ind
        neuron_connectors_as_pre = self.syn_data[ self.syn_data['pre_neuron_id'] == neuron_id]
        was = []
        for cid, pnid in zip(neuron_connectors_as_pre['connector_id'], neuron_connectors_as_pre['pre_node_id']):
            detector_label = f"{cid}det_on{pnid}"
            segment_id = node_to_segment[str(pnid)]
            
            if (cid, pnid) in was:
                continue
            was.append((cid, pnid))
            pos = random() # TODO: поменять на более реалистичные координаты
            decor.place(f"(on-components {pos} (segment {segment_id}))", arb.threshold_detector(-10 * arb.units.mV), detector_label)  # I have changed there pre-became detector and post-became synapse(by arbor documentation)
    
        
        neuron_connectors_as_post =  self.syn_data[ self.syn_data['post_neuron_id'] == neuron_id]
        was = []
        for cid, pnid in zip(neuron_connectors_as_post['connector_id'], neuron_connectors_as_post['post_node_id']):
            synapse_label = f"{cid}syn_on{pnid}"
            segment_id = node_to_segment[str(pnid)]
            if (cid, pnid) in was:
                continue
            was.append((cid, pnid))
            pos = random() # TODO: поменять на более реалистичные координаты
            decor.place(f"(on-components {pos} (segment {segment_id}))", arb.synapse(self.synt), synapse_label)
        return decor

    def connections_on(self, ind):
        to_neuron_id = ind
        connections = []
        neuron_connectors = self.syn_data[self.syn_data['post_neuron_id'] == to_neuron_id]

        for from_neuron_id, connector_id, pre_node, post_node in zip(
        neuron_connectors['pre_neuron_id'],
        neuron_connectors['connector_id'],
        neuron_connectors['pre_node_id'],
        neuron_connectors['post_node_id']
    ):
            from_gid = self.id_to_gid.get(from_neuron_id, None)
            if from_gid is None:
                print(from_neuron_id, 'is not in the given network')
                continue
            source_label = f"{connector_id}det_on{pre_node}"
            target_label = f"{connector_id}syn_on{post_node}"

            cnn = arb.connection(
                (from_gid, source_label),  # source: pre cell + detector label
                target_label,              # target synapse label on THIS cell
                0.1,                    # weight
                0.1 * arb.units.ms      # delay
            )
            connections.append(cnn)
        return connections

    def run(self, ind):
        gid = self.id_to_gid[ind]
        print(gid, 'is', ind)
        # --- дерево ---
        tree = arb.segment_tree()
        tp = self.graph[ind]
        root_candidates = [n for n, d in tp.nodes(data=True) if d['type'] == 'root']
        if len(root_candidates) != 1:
            raise Exception(f"В нейроне {ind} найдено {len(root_candidates)} узлов с тегом root")
        root_node = root_candidates[0]

        # Добавляем корневой сегмент
        soma_id = tree.append(parent=arb.mnpos, prox=arb.mpoint(*self.__get_node_geometry(root_node)),
                            dist=arb.mpoint(*self.__get_node_geometry(root_node)), tag=int(root_node))
        node_to_segment = {root_node:soma_id}

        # Рекурсивно добавляем остальные сегменты
        self.__in_graph(soma_id, root_node, node_to_segment, tree, tp)
        print(ind ,'tree - finished')
        

        # --- декоратор ---
        decor = self.decorator(ind, node_to_segment)
        print(ind ,'decorator - finished')
        
        # --- connectors ---
        connections = self.connections_on(ind)
        print(ind ,'connectors - finished')
        
        path = mk_dir(self.output_path, f"{gid}")
        name_morphology = os.path.join(path, f"morphology.arbc")
        name_decor = os.path.join(path, f"decor.arbc")
        name_mapping = os.path.join(path, f"mapping.json")
        name_connectors = os.path.join(path, f"connectors.pickle")
        arb.write_component(arb.morphology(tree), name_morphology)
        arb.write_component(decor, name_decor)
        with open(name_connectors, "wb") as f:
            pickle.dump(connections, f)
        with open(name_mapping, "w") as f:
            json.dump(node_to_segment, f)
        print(ind ,'finished')