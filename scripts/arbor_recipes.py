#arbor_recipies.py
import arbor as arb
from data_preparation import simulation_context
import networkx as nx
from pathlib import Path
import os
from random import random
import json
import pickle
# structure: [neuron_id][node_id][iclamp_list]
# arb.iclamp(100 * arb.units.ms, 10 * arb.units.ms, 40 * arb.units.nA)
class basic_recipe(arb.recipe):

    def __init__(self,
                sc:simulation_context,
                record_soma = True,
                nodes_to_record = None,
                iclamp_schedule:dict[int:dict[[str|int]:list[arb.iclamp]]] = None,
                cable_type = 'hh',
                synapce_type = 'expsyn',
                neurite_params=None, #params(pandas data) for each neurite 
    ):
        '''
        Базовый класс с hh на всех участках, expsyn и трешхолд детекторами
        никак не учитывает удаленные ноды
        '''
        super().__init__()
        self.iclamp_schedule:dict[int:dict[[str|int]:list[arb.iclamp]]] = {} if iclamp_schedule is None else iclamp_schedule
        self.gids = {v:k for k, v in enumerate(sc.neurons)}
        self.paths = [sc.path_to_gml(neuron_ind) for neuron_ind in sc.neurons]  
        self.ptog = {int(Path(p).stem):g for g, p in enumerate(self.paths)}
        self.gtoi = {g:int(Path(p).stem) for g, p in enumerate(self.paths)}
        self.ncells = len(self.paths)
        self.nodes_to_record = [] if nodes_to_record is None else nodes_to_record
        self.n_to_arb_coords = dict()
        self.synt = synapce_type
        self.cablet = cable_type

        
        self.record_soma = record_soma

        self.syn_data = sc.synaptic_table
        self.node_metadata = sc.node_metadata

        
        if neurite_params is not None:
            # Expect a column "neuron_id"
            self.neurite_params = neurite_params.set_index("neuron_id")
        else:
            self.neurite_params = None



    def __get_node_geometry(self, ind):
        '''x, y, z, r'''
        ind = int(ind)
        s = self.node_metadata[self.node_metadata['node_id'] == ind]
        return s.to_numpy()[0, 1:5]
    
    def __in_graph(self, parent_segment_id, current_node, node_to_segment, tree, tp, **kwargs):
        gid = kwargs['gid']
        for to_ in tp.predecessors(current_node):
            if tp.nodes[to_].get('type') == 'connector':
                continue
            segment_id = tree.append(parent_segment_id, arb.mpoint(*self.__get_node_geometry(current_node)),
                                    arb.mpoint(*self.__get_node_geometry(to_)), tag=int(to_))
            node_to_segment[to_] = segment_id
            if int(to_) in self.nodes_to_record:
                if gid not in self.n_to_arb_coords:
                    self.n_to_arb_coords[gid] = dict()
                self.n_to_arb_coords[gid][int(to_)] = segment_id
            self.__in_graph(segment_id, to_, node_to_segment, tree, tp, **kwargs)

    def num_cells(self):
        return self.ncells

    def cell_kind(self, gid):
        return arb.cell_kind.cable

    def cell_description(self, gid):
        print(gid)

        #self.ptog[int(Path(self.paths[gid]).stem)] = gid
        #self.gtoi[gid] = int(Path(self.paths[gid]).stem)

        tp: nx.Graph = nx.read_gml(self.paths[gid])
        tree = arb.segment_tree()
        root_candidates = [n for n, d in tp.nodes(data=True) if d['type'] == 'root']
        if len(root_candidates) != 1:
            raise Exception(f"В нейроне {self.paths[gid]} найдено {len(root_candidates)} узлов с тегом root")
        root_node = root_candidates[0]

        # Словарь для хранения соответствия node_id -> segment_id
        node_to_segment = {}

        # Добавляем корневой сегмент
        soma_id = tree.append(parent=arb.mnpos, prox=arb.mpoint(*self.__get_node_geometry(root_node)),
                            dist=arb.mpoint(*self.__get_node_geometry(root_node)), tag=int(root_node))
        node_to_segment[root_node] = soma_id

        # Рекурсивно добавляем остальные сегменты
        self.__in_graph(soma_id, root_node, node_to_segment, tree, tp, gid = gid)

        # Создаем decor и labels
        decor = arb.decor()
        neuron_id = self.gtoi[gid]
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


        neuron_id = self.gtoi[gid]
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
            

        
        if neuron_id in self.iclamp_schedule:
            for k, v in self.iclamp_schedule[neuron_id].items(): # There should be used neuron_id insead gid
                if k == 'soma':
                    k = 0
                assert type(k) is int
                assert isinstance(v, arb.iclamp)
                decor.place(f"(on-components 0.5 (segment {node_to_segment[k]}))", v, f'ic{k}')

        cc = arb.cable_cell(tree, decor)
        return cc

    def connections_on(self, gid):
        #TODO надо разобратся откуда берутся дубликаты в synaptic table. 
        #По идеи это вызвано тем что мы храним это как ребра для графа, а не для метаграфа, по этому сейчас все должно быть правильно по идеи
        to_neuron_id = self.gtoi[gid]
        connections = []
        neuron_connectors = self.syn_data[self.syn_data['post_neuron_id'] == to_neuron_id] #By arbor documentation there should be returned all incoming connections

        for from_neuron_id, connector_id, pre_node, post_node in zip(
        neuron_connectors['pre_neuron_id'],
        neuron_connectors['connector_id'],
        neuron_connectors['pre_node_id'],
        neuron_connectors['post_node_id']
    ):
            from_gid = self.ptog[from_neuron_id]
            source_label = f"{connector_id}det_on{pre_node}"
            target_label =   f"{connector_id}syn_on{post_node}"

            cnn = arb.connection(
                (from_gid, source_label),  # source: pre cell + detector label
                target_label,              # target synapse label on THIS cell
                0.1,                    # weight
                0.1 * arb.units.ms      # delay
            )
            connections.append(cnn)
        return connections

    def global_properties(self, kind):
        return arb.neuron_cable_properties()
    
    def probes(self, gid):
        somas = [arb.cable_probe_membrane_voltage("(on-components 0.0 (segment 0))", 'soma')] if self.record_soma else [] 
        other_segments = [arb.cable_probe_membrane_voltage(f"(on-components 0.5 (segment {self.n_to_arb_coords[gid][nr]}))", f'node{nr}') for nr in self.nodes_to_record] if gid in self.n_to_arb_coords else []
        return somas + other_segments



class optimized_recipe(arb.recipe):
    def __init__(self, connectome_dir,
                record_soma = True,
                nodes_to_record = None,
                iclamp_schedule:dict[int:dict[[str|int]:list[arb.iclamp]]] = None,
    ):
        '''
        загружает уже предопределенные компоненты, добавляя к ним контекстные свойства
        '''
        super().__init__()
        self.iclamp_schedule:dict[int:dict[[str|int]:list[arb.iclamp]]] = {} if iclamp_schedule is None else iclamp_schedule
        self.nodes_to_record = [] if nodes_to_record is None else nodes_to_record
        self.record_soma = record_soma
        self.connectome_dir = connectome_dir
        with open(os.path.join(self.connectome_dir, 'gid_mapping.json'), 'r') as file:
            self.mapping = json.load(file)
        
        self.gid_to_neuron_id = {v:int(k) for k, v in self.mapping.items()}

        self.ncells = len(self.mapping)

    def num_cells(self):
        return self.ncells

    def cell_kind(self, gid):
        return arb.cell_kind.cable

    def cell_description(self, gid):
        print(gid)

        # тут мы загружаем все компоненты
        gp = os.path.join(self.connectome_dir, str(gid))
        pd = os.path.join(gp, 'decor.arbc')
        pt = os.path.join(gp, 'morphology.arbc')
        pm = os.path.join(gp, 'mapping.json')
        decor = arb.load_component(pd).component
        morphology = arb.load_component(pt).component
        with open(pm, 'r') as file:
            node_to_segment = json.load(file)
        neuron_id = self.gid_to_neuron_id[gid]
        
        # тут нужно к уже существующему декоратору `decor`, добавить iclamp

        if neuron_id in self.iclamp_schedule:
            for k, v in self.iclamp_schedule[neuron_id].items():
                if k == 'soma':
                    k = 0
                assert type(k) is int
                assert isinstance(v, arb.iclamp)
                decor.place(f"(on-components 0.5 (segment {node_to_segment[k]}))", v, f'ic{k}')

        # тут собраем cable_cell
        cc = arb.cable_cell(morphology, decor)
        return cc

    def connections_on(self, gid):
        gp = os.path.join(self.connectome_dir, str(gid))
        pc = os.path.join(gp, 'connectors.pickle')
        with open(pc, 'rb') as f:
            connections = pickle.load(f)
        return connections

    def global_properties(self, kind):
        return arb.neuron_cable_properties()
    
    def probes(self, gid):
        #TODO это нужно переделать
        #somas = [arb.cable_probe_membrane_voltage("(on-components 0.0 (segment 0))", 'soma')] if self.record_soma else [] 
        #other_segments = [arb.cable_probe_membrane_voltage(f"(on-components 0.5 (segment {self.n_to_arb_coords[gid][nr]}))", f'node{nr}') for nr in self.nodes_to_record] if gid in self.n_to_arb_coords else []
        #return somas + other_segments
        return []