import arbor as arb
from data_preparation import simulation_context
import networkx as nx
from pathlib import Path
import os
from random import random
# structure: [neuron_id][node_id][iclamp_list]
# arb.iclamp(100 * arb.units.ms, 10 * arb.units.ms, 40 * arb.units.nA)
class basic_recipe(arb.recipe):
    def __init__(self, sc:simulation_context, record_soma = True, nodes_to_record = None, iclamp_schedule:dict[int:dict[[str|int]:list[arb.iclamp]]] = None):
        '''
        Базовый класс с hh на всех участках, expsyn и трешхолд детекторами
        никак не учитывает удаленные ноды
        '''
        super().__init__()
        self.iclamp_schedule:dict[int:dict[[str|int]:list[arb.iclamp]]] = [] if iclamp_schedule is None else iclamp_schedule
        self.paths = [sc.path_to_neurons + neuron_path for neuron_path in os.listdir(sc.path_to_neurons)]
        self.gids = {int(Path(v).stem):k for k, v in enumerate(self.paths)}
        self.ptog = dict()
        self.gtoi = dict()
        self.ncells = len(self.paths)
        self.nodes_to_record = [] if nodes_to_record is None else nodes_to_record
        self.n_to_arb_coords = dict()

        
        self.record_soma = record_soma

        self.syn_data = sc.synaptic_table
        self.node_metadata = sc.node_metadata

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
            segment_id = tree.append(parent_segment_id, arb.mpoint(**self.get_node_geometry(current_node)),
                                    arb.mpoint(**self.get_node_geometry(to_)), tag=int(to_))
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

        self.ptog[int(Path(self.paths[gid]).stem)] = gid
        self.gtoi[gid] = int(Path(self.paths[gid]).stem)

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
                            dist=arb.mpoint(**self.__get_node_geometry(root_node)), tag=int(root_node))
        node_to_segment[root_node] = soma_id

        # Рекурсивно добавляем остальные сегменты
        self.__in_graph(soma_id, root_node, node_to_segment, tree, tp)

        # Создаем decor и labels
        decor = arb.decor()
        decor.paint('(all)', arb.density('hh'))

        neuron_id = self.gtoi[gid]
        neuron_connectors_as_pre = self.syn_data[ self.syn_data['pre_neuron_id'] == neuron_id]
        for cid, pnid in zip(neuron_connectors_as_pre['connector_id'], neuron_connectors_as_pre['pre_node_id']):
            pos = random() # TODO: поменять на более реалистичные координаты
            synapse_label = f"{cid}syn"
            segment_id = node_to_segment[str(pnid)]
            decor.place(f"(on-components {pos} (segment {segment_id}))", arb.synapse("expsyn"), synapse_label)
        
        neuron_connectors_as_post =  self.syn_data[ self.syn_data['post_neuron_id'] == neuron_id]
        for cid, pnid in zip(neuron_connectors_as_post['connector_id'], neuron_connectors_as_post['post_node_id']):
            pos = random() # TODO: поменять на более реалистичные координаты
            detector_label = f"{cid}det"
            segment_id = node_to_segment[str(pnid)]
            decor.place(f"(on-components {pos} (segment {segment_id}))", arb.threshold_detector(-10 * arb.units.mV), detector_label)

        if self.gtoi[gid] in self.iclamp_schedule:
            for k, v in self.iclamp_schedule[gid].items():
                if k == 'soma':
                    k = 0
                assert type(k) is int
                assert isinstance(v, arb.iclamp)
                decor.place(f"(on-components 0.5 (segment {node_to_segment[k]}))", v, f'ic{k}')

        cc = arb.cable_cell(tree, decor)
        return cc

    def connections_on(self, gid):
        to_neuron_id = self.gtoi[gid]
        connections = []
        neuron_connectors =  self.syn_data[ self.syn_data['pre_neuron_id'] == to_neuron_id]

        for to_N, connector_id in zip(neuron_connectors['post_neuron_id'], neuron_connectors['connector_id']):
            from_gid = self.ptog[to_N]
            source = (from_gid, f"{connector_id}det")
            target =  f"{connector_id}syn"
            cnn = arb.connection(source, target, 0.1, 0.1 * arb.units.ms)
            connections.append(cnn)
        return connections

    def global_properties(self, kind):
        return arb.neuron_cable_properties()
    
    def probes(self, gid):
        somas = [arb.cable_probe_membrane_voltage("(on-components 0.0 (segment 0))", 'soma')] if self.record_soma else [] 
        other_segments = [arb.cable_probe_membrane_voltage(f"(on-components 0.5 (segment {self.n_to_arb_coords[gid][nr]}))", f'node{nr}') for nr in self.nodes_to_record] if gid in self.n_to_arb_coords else []
        return somas + other_segments