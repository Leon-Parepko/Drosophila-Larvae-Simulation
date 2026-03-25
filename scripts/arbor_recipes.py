#arbor_recipies.py
import arbor as arb
import networkx as nx
from pathlib import Path
import os
from random import random
import json
import pickle
# iclamp dtype dict[int:dict[[str|int]:list[arb.iclamp]]]
class optimized_recipe(arb.recipe):
    def __init__(self, connectome_dir,
                record_soma = True,
                nodes_to_record = None,
                iclamp_schedule = None,
                neurite_params = None,
                cablet = 'hh'
    ):
        '''
        загружает уже предопределенные компоненты, добавляя к ним контекстные свойства
        '''
        super().__init__()
        self.neurite_params = neurite_params
        self.iclamp_schedule = {} if iclamp_schedule is None else iclamp_schedule
        self.nodes_to_record = [] if nodes_to_record is None else nodes_to_record
        self.record_soma = record_soma
        self.connectome_dir = connectome_dir
        with open(os.path.join(self.connectome_dir, 'gid_mapping.json'), 'r') as file:
            self.mapping = json.load(file)
        self.cablet = cablet
        
        self.gid_to_neuron_id = {v:int(k) for k, v in self.mapping.items()}

        self.ncells = len(self.mapping)

    def num_cells(self):
        return self.ncells

    def cell_kind(self, gid):
        return arb.cell_kind.cable

    def cell_description(self, gid):

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
        
        # тут нужно к уже существующему декоратору `decor`, добавить iclamp

        if neuron_id in self.iclamp_schedule:
            for k, v in self.iclamp_schedule[neuron_id].items():
                if k == 'soma':
                    q = 0
                else:
                    q = node_to_segment[k]
                assert type(q) is int
                assert isinstance(v, arb.iclamp)
                decor.place(f"(on-components 0.5 (segment {q}))", v, f'ic{k}')

        # тут собраем cable_cell
        cc = arb.cable_cell(morphology, decor)
        return cc

    def connections_on(self, gid):
        gp = os.path.join(self.connectome_dir, str(gid))
        pc = os.path.join(gp, 'connectors.json')
        with open(pc, 'r') as f:
            connections = json.load(f)
        return [arb.connection(
                source = tuple(c['source']),
                dest = c['target'],
                weight = c['weight'],
                delay = c['delay'] * arb.units.ms
            ) for c in connections]

    def global_properties(self, kind):
        return arb.neuron_cable_properties()
    
    def probes(self, gid):
        # а это и так прекрасно работает
        somas = [arb.cable_probe_membrane_voltage("(on-components 0.0 (segment 0))", 'soma')] if self.record_soma else [] 
        #TODO это нужно переделать
        #other_segments = [arb.cable_probe_membrane_voltage(f"(on-components 0.5 (segment {self.n_to_arb_coords[gid][nr]}))", f'node{nr}') for nr in self.nodes_to_record] if gid in self.n_to_arb_coords else []
        #return somas + other_segments
        return somas