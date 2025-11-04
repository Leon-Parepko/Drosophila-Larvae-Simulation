import os
import numpy as np
import pandas as pd
import networkx as nx
from collections import deque
from neuron import h, coreneuron
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px


h.load_file('stdrun.hoc')


class Network:
    def __init__(self, neuron_ids, neurons_dir='./Datasets/Original/neurons', meta_pkl='./Datasets/Original/Metadata(auto).pkl'):
        self.NEURONS = neuron_ids
        self.NEURONS_DIR = neurons_dir
        self.META_PKL = meta_pkl
        
        self.graphs = {}
        self.sections = {}
        self.somas = {}

        self.synapses = []
        self.t_vec = None
        self.v_records = {}
        
        self.stimulas = []

        print('number of neurons:', len(self.NEURONS), '\n')

    
    def __set_segments(self, section):
        try:
            # lambda_f - длина волны на 100 Hz
            lam = section.lambda_f(100)
        except:
            # or default
            lam = 100.0

        # считаем сколько нужно сегментов
        nseg = int(section.L / (0.1 * lam) + 0.999)

        # колво сегментов должно быть нечетным - в neuron уточнение
        if nseg % 2 == 0:
            nseg += 1

        # min - 1
        section.nseg = max(1, nseg)
        return
    
    
    def load_graphs(self, verbose=False):
        if verbose:
            print('loading graphs \n')

        for neuron_id in self.NEURONS:
            path = os.path.join(self.NEURONS_DIR, f'{neuron_id}.gml')

            if not os.path.exists(path):
                raise FileNotFoundError(f'{path} not found')

            # read graph
            graph = nx.read_gml(path)
            self.graphs[neuron_id] = graph

            # root - soma
            root_nodes = [n for n, data in graph.nodes(data=True)
                          if data.get('type') == 'root']

            if verbose:
                print(f"Neuron {neuron_id}   \t {graph.number_of_nodes()} nodes \t root={root_nodes[0] if root_nodes else 'NONE'}")

            
    def build_sections(self, verbose=False, soma_params=None, dendrite_params=None):
        
        # значения по умолчанию для сомы и дендритов
        default_soma_params = {
            'L': 20,             # длина
            'diam': 20,          # диаметр
            'Ra': 100,           # аксиальное сопротивление
            'cm': 1,             # емкость мембраны
            
            'gnabar_hh': 0.12,   # проводимость натриевых каналов
            'gkbar_hh': 0.036,   # проводимость калиевых каналов
            'gl_hh': 0.0003,     # leak проводимость
            'el_hh': -65.0
        }
        default_dendrite_params = {
            'L': 50.0,           # длина
            'diam': 1.0,         # диаметр
            'Ra': 100.0,         # аксиальное сопротивление
            'cm': 1.0,           # емкость мембраны
            
            'g_pas': 0.0001,     # пассивная проводимость
            'e_pas': -65.0       # reversal potential
        }
        
        # объединяем дефолтные параметры с переданными
        soma_params = {**default_soma_params, **(soma_params or {})}
        dendrite_params = {**default_dendrite_params, **(dendrite_params or {})}
        
        if verbose:
            print('building sections \n')
        
        for neuron_id, graph in self.graphs.items():
            # create soma (take root as soma)        
            for node_id, data in graph.nodes(data=True):
                if data.get('type') == 'root':
                    
                    # Use HH mechanism
                    soma = h.Section(name=f"soma_{neuron_id}")
                    soma.insert('hh')
                    
                    for key, value in soma_params.items():
                        setattr(soma, key, value)
                    self.__set_segments(soma)
                    self.sections[(neuron_id, node_id)] = soma
                    self.somas[neuron_id] = soma
                    break

            # create dendrites
            dendrite_count = 0
            for node_id, data in graph.nodes(data=True):
                node_type = data.get('type')
                if node_type in ('slab', 'branch', 'end'):
                    
                    # Use passive mechanism
                    dend = h.Section(name=f"dend_{neuron_id}_{node_id}")              
                    dend.insert('pas')
                    
                    for key, value in dendrite_params.items():
                        setattr(dend, key, value)
                    self.__set_segments(dend)
                    self.sections[(neuron_id, node_id)] = dend
                    dendrite_count += 1
                    
            if verbose:
                print(f'{neuron_id}: {dendrite_count} dendrites') 
        return

                
    def connect_morphology(self, verbose=False):
        if verbose:
            print('connecting morphology \n')

        for neuron_id, graph in self.graphs.items():

            morph_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') in ('root', 'slab', 'branch', 'end')]

            # подграф только из этих узлов
            subgraph = graph.subgraph(morph_nodes).to_undirected()

            # находим root
            root = None
            for node in morph_nodes:
                if graph.nodes[node].get('type') == 'root':
                    root = node
                    break

            if not root:
                continue

            # BFS от сомы
            visited = {root}  # visisted nodes
            queue = deque([root])
            connections = 0

            while queue:
                parent_node = queue.popleft()

                for child_node in subgraph.neighbors(parent_node):
                    # if visited -> skip
                    if child_node in visited:
                        continue

                    # get the Neuron section
                    parent_sec = self.sections.get((neuron_id, parent_node))
                    child_sec = self.sections.get((neuron_id, child_node))

                    if parent_sec and child_sec:
                        # соединяем начало ребенка (0) к концу родителя (1)
                        child_sec.connect(parent_sec(1), 0)

                        # отмечаем как посещенный
                        visited.add(child_node)

                        # добавляем в очередь
                        queue.append(child_node)

                        # Увеличиваем счетчик
                        connections += 1

            if verbose:
                print(f'{neuron_id}: {connections} connections')
        return
    
                
    def build_synapses(self, verbose=False, synapse_params=None, netcon_params=None):
        
        default_synapse_params = {
            'tau1': 0.5,  # время нарастания
            'tau2': 2.0,  # время спада
            'e': 0.0  # reversal potential (возбуждающий, reversal ~ 0 mV)
        }

        default_netcon_params = {
            'threshold': 0.0,  # порог (мВ) - спайк когда V > 0
            'weight': 0.01,  # вес синапса (микросименсы)
            'delay': 1.0  # задержка (ms)
        }

        # объединяем дефолтные параметры с переданными
        synapse_params = {**default_synapse_params, **(synapse_params or {})}
        netcon_params = {**default_netcon_params, **(netcon_params or {})}

        if verbose:
            print('building synapses \n')

        meta = pd.read_pickle(self.META_PKL)
        meta = meta.set_index('skeleton_id')
        meta.index = meta.index.astype(str)

        connector_tables = {}

        for neuron_id in self.NEURONS:
            # check if we have it in metadata
            if neuron_id not in meta.index:
                continue

            # get the connectors table for exact skeleton id
            conn_df = meta.loc[neuron_id, 'connectors']

            df = conn_df[['connector_id', 'node_id', 'type']].copy()

            # converting types
            df['connector_id'] = df['connector_id'].astype(int)
            df['node_id'] = df['node_id'].astype(str)

            # 0 - пресинаптик
            # 1 - постсинаптик
            df['bit'] = df['type'].astype(str).str[-1].astype(int)

            connector_tables[neuron_id] = df
            if verbose:
                print(f'{neuron_id}: {len(df)} connectors')

        synapse_count = 0

        for i, neuron_a in enumerate(self.NEURONS):
            for neuron_b in self.NEURONS[i + 1:]:

                if neuron_a not in connector_tables or neuron_b not in connector_tables:
                    continue

                # merge таблицы по connector_id
                merged = connector_tables[neuron_a].merge(connector_tables[neuron_b], on='connector_id', suffixes=('_a', '_b'))

                # если нет общих коннекторов - пропускаем
                if len(merged) == 0:
                    continue

                if verbose:
                    print(f'{neuron_a} <-> {neuron_b}: {len(merged)} shared')

                # Для каждого общего коннектора
                for _, row in merged.iterrows():
                    # получаем 0/1 для каждого нейрона
                    bit_a = int(row['bit_a'])
                    bit_b = int(row['bit_b'])
                    node_a = row['node_id_a']
                    node_b = row['node_id_b']

                    # направление синапса
                    # если A=0 и B=1, то A -> B
                    if bit_a == 0 and bit_b == 1:
                        pre_neuron = neuron_a  # отсюда идет сигнал
                        pre_node = node_a # узел от куда берем
                        post_neuron = neuron_b  # сюда приходит
                        post_node = node_b  # узел где синапс

                    # если A=1 и B=0, то B -> A
                    elif bit_a == 1 and bit_b == 0:
                        pre_neuron = neuron_b
                        pre_node = node_b
                        post_neuron = neuron_a
                        post_node = node_a

                    # если оба 0 или оба 1 - пропускаем
                    else:
                        continue
                    
                    if verbose:
                        print(f"pre neuron {pre_neuron} ({pre_node}) \t -> \t post neuron {post_neuron} ({post_node})")
                    
                    post_section = self.sections.get((post_neuron, post_node))
                    if not post_section:
                        continue

                    pre_section = self.sections.get((pre_neuron, pre_node))
                    if not pre_section:
                        continue

                    # создаем синапс на постсинаптическом дендрите
                    synapse = h.Exp2Syn(post_section(0.5))  # в центре секции
                    
                    # Synapse params
                    for key, value in synapse_params.items():
                        setattr(synapse, key, value)

                    # создаем детектор спайков - voltage сомы пресинаптика
                    netcon = h.NetCon(
                        pre_section(0.5)._ref_v,  # откуда берем voltage
                        synapse,                  # куда отправляем event
                        sec=pre_section
                    )

                    # NetCon Params
                    netcon.threshold = netcon_params['threshold']
                    netcon.weight[0] = netcon_params['weight']
                    netcon.delay = netcon_params['delay']

                    # Save synapse and its netcon
                    self.synapses.append([synapse, netcon])
                    synapse_count += 1

        if verbose:
            print(f'Synapses created: {synapse_count}')
            
        return

        
            
    def setup_recording(self, neurons=[], sections=[], verbose=False):
        if verbose:
            print('setting recordings \n')
        # вектор для записи времени
        self.t_vec = h.Vector().record(h._ref_t)

        if len(sections) > 0:
            for section_id in sections:
                # Filter self.sections to find the section that matches the given ID
                section = next((sec for (neuron_id, node_id), sec in self.sections.items() if node_id == section_id), None)
                if section:
                    vec = h.Vector().record(section(0.5)._ref_v)
                    self.v_records[section_id] = vec
                    
        elif len(neurons) > 0:
            for neuron_id in neurons:
                soma = self.somas.get(neuron_id)
                if soma:
                    vec = h.Vector().record(soma(0.5)._ref_v)
                    self.v_records[neuron_id] = vec
        else:
            for neuron_id, soma in self.somas.items():
                vec = h.Vector().record(soma(0.5)._ref_v)
                self.v_records[neuron_id] = vec

                
    def setup_stimulus(self, neurons=None, sections=None, start_time=0, duration=100, amplitude=0.1, verbose=False):
        if verbose:
            print('adding stimulus \n')
        if sections:
            for section_id in sections:
                section = next((sec for (neuron_id, node_id), sec in self.sections.items() if node_id == section_id), None)
                if section:
                    stim = h.IClamp(section(0.5))  
                    stim.delay = start_time  
                    stim.dur = duration  
                    stim.amp = amplitude  
                    if verbose:
                        print(f'Stimulating section {section_id}: {stim.amp} nA for {stim.dur} ms')
        elif neurons:
            for neuron_id in neurons:
                soma = self.somas[neuron_id]
                if soma:
                    stim = h.IClamp(soma(0.5))  
                    stim.delay = start_time  
                    stim.dur = duration  
                    stim.amp = amplitude 
                    self.stimulas.append(stim)
                    if verbose:
                        print(f'Stimulating neuron {neuron_id}: {stim.amp} nA for {stim.dur} ms')
        else:
            print("No neurons or sections provided for stimulation.")
            return
    

    def run(self, duration=100, dt=0.025, steps_per_ms=40, verbose=False):

        if verbose:
            print(f'running simulation for {duration}\n')

        h.dt = dt  # шаг времени
        h.steps_per_ms = steps_per_ms  # количество шагов на миллисекунду
        h.v_init = -70  # начальный voltage

        h.finitialize(-70)

        # Run simulation
        h.continuerun(duration)

        if verbose:
            print('simulation done \n')

        t = list(self.t_vec)
        voltages = {nid: list(vec) for nid, vec in self.v_records.items()}

        return t, voltages

    def analyze(self, t, voltages):
        print('ANALYZATION \n')

        print("Spikes crossing 0 mV:")
        for neuron_id, v in voltages.items():
            v_array = np.array(v)
            crossings = np.where(v_array > 0)[0]

            if len(crossings) > 0:
                spike_time = t[crossings[0]]
                print(f'{neuron_id}: {len(crossings)} SPIKES at (TODO) {spike_time}')
            else:
                print('{neuron_id}: no spike')

        # максимальная деполяризацию
        print('Maximum depolirazation')
        for neuron_id, v in voltages.items():
            v_array = np.array(v)
            baseline = v_array[0]  # начальное значение
            max_v = np.max(v_array)  # максимум
            depol = max_v - baseline  # разница
            print(f'{neuron_id}: {depol} mV')
            
            
    def reset(self):
        h.finitialize(-70)
        h.t = 0
        self.t_vec = None
        self.v_records = {}
        self.stimulas = []
        return


    def plot_results_3d(self, t, voltages, interactive=False):
        # Use plotly if interactive     
        if interactive:
            neuron_ids = list(voltages.keys())
            cmap = px.colors.sequential.Viridis
            n_colors = len(cmap)

            fig = go.Figure()

            for i, neuron_id in enumerate(neuron_ids):
                v = voltages[neuron_id]
                color = cmap[int(i / max(1, len(neuron_ids)-1) * (n_colors-1))]
                fig.add_trace(go.Scatter3d(
                    x=t, 
                    y=[i]*len(t),
                    z=v,
                    mode='lines',
                    name=str(neuron_id),
                    line=dict(color=color, width=4)
                ))

            fig.update_layout(
                title='Network Activity (3D View)',
                scene=dict(
                    xaxis_title='Time (ms)',
                    yaxis_title='Neuron ID',
                    zaxis_title='Voltage (mV)',
                ),
                template='plotly_dark',
                width=1000,
                height=700,
                showlegend=True
            )

            fig.show()
            return

        # Use matplotlib if non-interactive
        else:
            fig = plt.figure(figsize=(12, 7))
            ax = fig.add_subplot(111, projection='3d')

            cmap = cm.get_cmap('viridis', len(voltages))

            for i, (neuron_id, v) in enumerate(voltages.items()):
                color = cmap(i)
                # создаём массив нейронных индексов той же длины, что и t
                y = [i] * len(t)
                ax.plot(t, y, v, color=color, linewidth=2, label=str(neuron_id))

            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Neuron ID')
            ax.set_zlabel('Voltage (mV)')
            ax.set_title('Network Activity (3D view)')

            # перемещаем легенду за пределы графика
            ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
            plt.tight_layout()
            plt.show()
            return



    def plot_results(self, t, voltages, interactive=False):
        # Use plotly if interactive  
        if interactive:
            # используем градиент Viridis из sequential палитры
            colors = px.colors.sequential.Viridis
            fig = go.Figure()

            neuron_ids = list(voltages.keys())
            n_colors = len(colors)

            for i, (neuron_id, v) in enumerate(voltages.items()):
                color = colors[i % n_colors]
                fig.add_trace(go.Scatter(
                    x=t,
                    y=v,
                    mode='lines',
                    name=str(neuron_id),
                    line=dict(color=color, width=2)
                ))

            fig.update_layout(
                title='Network Activity',
                xaxis_title='Time (ms)',
                yaxis_title='Voltage (mV)',
                template='plotly_dark',
                hovermode='x unified',
                width=1000,
                height=600
            )

            # вертикальная линия при 10 мс
            fig.add_vline(x=10, line_dash='dash', line_color='white', opacity=0.5)

            fig.show()
            return

        # Use matplotlib if non-interactive
        else:
            plt.figure(figsize=(12, 6))

            cmap = cm.get_cmap('viridis', len(voltages))

            for i, (neuron_id, v) in enumerate(voltages.items()):
                color = cmap(i)
                plt.plot(t, v, label=str(neuron_id), color=color, linewidth=2)

            plt.xlabel('Time (ms)')
            plt.ylabel('Voltage (mV)')
            plt.title('Network Activity')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axvline(x=10, color='black', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show() 
            return