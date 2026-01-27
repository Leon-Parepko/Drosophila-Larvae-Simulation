import os
import numpy as np
import pandas as pd
import networkx as nx
from collections import deque
from neuron import h, coreneuron
from mpl_toolkits.mplot3d import Axes3D
from tqdm.notebook import tqdm


h.load_file('stdrun.hoc')


class Network:
    def __init__(self, 
                 neuron_ids, 
                 neurons_dir='./Datasets/Original/neurons',
                 meta_pkl='./Datasets/Original/Metadata(auto).pkl', 
                 verbose=False):
        """
        Initialize the Network object, load neuron IDs and prepare containers for graphs,
        morphology sections, somas, synapses, recordings, and stimuli.
        
        Parameters
        ----------
        neuron_ids :  list - List of neuron identifiers to load and simulate.
        neurons_dir : str  - Path to directory containing GML morphology files.
        meta_pkl :    str  - Path to metadata .pkl file with connector information.
        verbose :     bool - If True — prints debug information during initialization.
        """
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

        if verbose:
            print('number of neurons:', len(self.NEURONS), '\n')


    
    def __set_segments(self, 
                       section):
        """
        Automatically compute and assign the number of spatial segments (nseg) 
        for a NEURON section based on the λ-rule for stable numerical simulation.

        Parameters
        ----------
        section : h.Section - NEURON section object for which nseg is computed.

        Returns
        -------
        None
        """
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
    

    
    def load_graphs(self, 
                    verbose=False, 
                    allow_tqdm=False):
        """
        Load neuron morphology graphs (.gml) into memory and store them inside the Network.
        Each graph contains morphology nodes with annotated types (root, slab, branch, end).

        Parameters
        ----------
        verbose :    bool - If True — prints information about each loaded neuron.
        allow_tqdm : bool - If True — uses tqdm progress bar for loading.

        Returns
        -------
        None
        """
        neurons_iter = self.NEURONS
        if verbose or allow_tqdm:
            neurons_iter = tqdm(self.NEURONS, desc="loading graphs")
    
        for neuron_id in neurons_iter:
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

        # TODO Rewrite to use this file
        # data = np.load("./Datasets/Original/neurons.npz", allow_pickle=True)
        # graphs = data["graphs"].item()
        
        # graphs[11279244]
        return


    
    def __normalize_params(self, 
                           default_params, 
                           user_params):
        """
        Convert user-provided mechanism parameters into a unified internal format.

        Accepts:
        - dict → global parameters for all neurons
        - pandas.DataFrame → per-neuron parameters (must contain 'neuron_id')
        - None → return default global parameters

        Parameters
        ----------
        default_params : dict                             - Default mechanism parameters.
        user_params :    dict or pandas.DataFrame or None - User-specified parameter override.

        Returns
        -------
        dict - Structure describing whether parameters are global or per-neuron.
        """
        
        # 1) Если None → использовать дефолт
        if user_params is None:
            return {"type": "global", "data": default_params}
    
        # 2) Если это dict → тоже глобальные параметры
        if isinstance(user_params, dict):
            merged = {**default_params, **user_params}
            return {"type": "global", "data": merged}
    
        # 3) Если это DataFrame → параметры по neuron_id
        if isinstance(user_params, pd.DataFrame):
            if "neuron_id" not in user_params.columns:
                raise ValueError("DataFrame with parameters must contain column 'neuron_id'")
    
            per_neuron = (
                user_params
                .set_index("neuron_id")
                .to_dict(orient="index")
            )
            return {"type": "per_neuron", "data": per_neuron}
    
        raise TypeError("soma_params/dendrite_params must be dict, DataFrame, or None")



    def build_sections(self, 
                       verbose=False, 
                       allow_tqdm=False, 
                       soma_mechanism='hh', 
                       soma_params=None, 
                       dendrite_mechanism='pas', 
                       dendrite_params=None):

        """
        Create NEURON Sections (soma and dendrites) for every neuron based on the 
        morphology graphs and insert the selected membrane mechanisms.

        Parameters
        ----------
        verbose :            bool - If True — prints number of created dendrites per neuron.
        allow_tqdm :         bool - If True — uses tqdm for progress visualization.
        soma_mechanism :     str  - Name of the mechanism inserted into soma sections (e.g., 'hh', 'pas').
        soma_params :        dict or pandas.DataFrame or None - Parameters for soma sections; can be global or per-neuron.
        dendrite_mechanism : str  - Mechanism name for dendritic sections.
        dendrite_params :    dict or pandas.DataFrame or None - Dendritic mechanism parameters; global or per-neuron.

        Returns
        -------
        None
        """
        
        # значения по умолчанию для сомы и дендритов
        if soma_mechanism == 'pas':
            pass # TODO
        
        elif soma_mechanism == 'hh':
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
            
        if dendrite_mechanism == 'pas':
            default_dendrite_params = {
                'L': 50.0,           # длина
                'diam': 1.0,         # диаметр
                'Ra': 100.0,         # аксиальное сопротивление
                'cm': 1.0,           # емкость мембраны

                'g_pas': 0.0001,     # пассивная проводимость
                'e_pas': -65.0       # reversal potential
            }
            
        elif dendrite_mechanism == 'hh':
            default_dendrite_params = {
                'L': 50.0,           # длина
                'diam': 1.0,         # диаметр
                'Ra': 100.0,         # аксиальное сопротивление
                'cm': 1.0,           # емкость мембраны

                'gnabar_hh': 0.12,   # проводимость натриевых каналов
                'gkbar_hh': 0.036,   # проводимость калиевых каналов
                'gl_hh': 0.0001,     # leak проводимость
                'el_hh': -65.0
            }

        # объединяем дефолтные параметры с переданными
        soma_cfg = self.__normalize_params(default_soma_params, soma_params)
        dend_cfg = self.__normalize_params(default_dendrite_params, dendrite_params)

        neurons_iter = self.graphs.items()
        if verbose or allow_tqdm:
            neurons_iter = tqdm(self.graphs.items(), desc="building sections")
        
        for neuron_id, graph in neurons_iter:
            # ========= SOMA =========
            # (take root as soma) 
            for node_id, data in graph.nodes(data=True):
                if data.get('type') == 'root':
    
                    soma = h.Section(name=f"soma_{neuron_id}")
                    soma.insert(soma_mechanism)
    
                    # Configure soma params
                    if soma_cfg["type"] == "global":
                        params = soma_cfg["data"]
                    else:
                        params = soma_cfg["data"].get(neuron_id, default_soma_params)
    
                    for k, v in params.items():
                        setattr(soma, k, v)
    
                    self.__set_segments(soma)
                    self.sections[(neuron_id, node_id)] = soma
                    self.somas[neuron_id] = soma
                    break

            # ========= DENDRITES =========
            dend_count = 0
    
            for node_id, data in graph.nodes(data=True):
                if data.get('type') in ('slab', 'branch', 'end'):
    
                    dend = h.Section(name=f"dend_{neuron_id}_{node_id}")
                    dend.insert(dendrite_mechanism)
    
                    # Configure dendrite params
                    if dend_cfg["type"] == "global":
                        params = dend_cfg["data"]
                    else:
                        params = dend_cfg["data"].get(neuron_id, default_dendrite_params)
    
                    for k, v in params.items():
                        setattr(dend, k, v)
    
                    self.__set_segments(dend)
                    self.sections[(neuron_id, node_id)] = dend
    
                    dend_count += 1
                    
            if verbose:
                print(f'{neuron_id}: {dend_count} dendrites') 
        return



    def connect_morphology(self, 
                           verbose=False, 
                           allow_tqdm=False):
        """
        Connect NEURON Sections according to the topology of morphology graphs using BFS.
        Ensures correct parent–child connectivity for soma and dendritic tree.

        Parameters
        ----------
        verbose :    bool - If True — prints number of created connections.
        allow_tqdm : bool - If True — uses tqdm progress bar.
        
        Returns
        -------
        None
        """
        
        neurons_iter = self.graphs.items()
        if verbose or allow_tqdm:
            neurons_iter = tqdm(self.graphs.items(), desc="connecting morphology")
        
        for neuron_id, graph in neurons_iter:

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
    

        
    def build_synapses(self, 
                       verbose=False, 
                       allow_tqdm=False, 
                       synapse_params=None, 
                       netcon_params=None):
        """
        Build synaptic connections between neurons using metadata from a .pkl file.
        Synapses are created whenever two neurons share a connector ID but with opposite
        pre/post roles.

        Parameters
        ----------
        verbose :        bool         - If True — prints debug information for each neuron pair.
        allow_tqdm :     bool         - If True — uses tqdm for progress visualization.
        synapse_params : dict or None - Parameters for Exp2Syn synapse objects (tau1, tau2, e).
        netcon_params :  dict or None - Parameters for NetCon objects (threshold, weight, delay).

        Returns
        -------
        None
        """
        
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



        neurons_iter = enumerate(self.NEURONS)
        if verbose or allow_tqdm:
            neurons_iter = tqdm(enumerate(self.NEURONS), desc="building synapses", total=len(self.NEURONS))
        
        for i, neuron_a in neurons_iter:
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

        
            
    def setup_recording(self, 
                        neurons=[], 
                        sections=[], 
                        verbose=False):
        """
        Configure NEURON recording vectors for time and membrane voltages.
        Allows recording either from selected neurons or specific morphology sections.

        Parameters
        ----------
        neurons :  list - Neuron IDs whose soma voltages must be recorded.
        sections : list - Section node-IDs to record membrane potential from.
        verbose :  bool - If True — prints debug information.

        Returns
        -------
        None
        """
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

        return


                
    def setup_stimulus(self, 
                       neurons=None, 
                       sections=None, 
                       start_time=0, 
                       duration=100, 
                       amplitude=0.1, 
                       verbose=False):

        """
        Insert IClamp stimuli into selected neuron somas or specific sections.

        Parameters
        ----------
        neurons :    list or None - IDs of neurons to stimulate.
        sections :   list or None - IDs of sections to apply stimulus to.
        start_time : float - Time in ms when stimulus begins.
        duration :   float - Duration of the injected current (ms).
        amplitude :  float - Amplitude of the IClamp current (nA).
        verbose :    bool  - If True — prints information about created stimuli.

        Returns
        -------
        None
        """
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
    


    def run(self, 
            duration=100, 
            dt=0.025, 
            steps_per_ms=40, 
            verbose=False):
        """
        Run a NEURON simulation for a specified duration and temporal resolution.

        Parameters
        ----------
        duration :     float - Total simulation duration (ms).
        dt :           float - Integration time step (ms).
        steps_per_ms : int   - Number of internal NEURON steps per millisecond.
        verbose :      bool  - If True — prints simulation status messages.

        Returns
        -------
        t : list - Simulation time points.
        voltages : dict - Recorded voltage traces per neuron or section.
        """
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



    def analyze(self, 
                t, 
                voltages, 
                spike_threshold=0, 
                verbose=False):
        """
        Detect spikes in simulated membrane voltages by finding local maxima 
        above a given threshold. Returns spike times and amplitudes.

        Parameters
        ----------
        t :               list  - Time points of the simulation.
        voltages :        dict  - Dict mapping IDs → voltage time series.
        spike_threshold : float - Minimum voltage for detecting a spike peak.
        verbose :         bool  - If True — prints spike statistics per neuron.

        Returns
        -------
        spike_times :      dict - Detected spike times for each neuron/section.
        spike_amplitudes : dict - Amplitudes of detected spikes.
        """
    
        spike_times = {}
        spike_amplitudes = {}
    
        # проверка — lengths must match per neuron
        T = len(t)
    
        for neuron_id, v in voltages.items():
    
            if len(v) != T:
                raise ValueError(
                    f"Length mismatch for neuron {neuron_id}: "
                    f"len(t)={T}, len(v)={len(v)}"
                )
    
            times = []
            amps = []
            n = len(v)
    
            # локальные пики
            for i in range(1, n - 1):
                if v[i] > v[i - 1] and v[i] > v[i + 1] and v[i] > spike_threshold:
                    times.append(t[i])
                    amps.append(v[i])
    
            spike_times[neuron_id] = times
            spike_amplitudes[neuron_id] = amps
    
            if verbose:
                if len(amps) == 0:
                    print(f"Neuron {neuron_id}: spikes=0")
                else:
                    print(
                        f"Neuron {neuron_id}: spikes={len(amps)}, "
                        f"mean={sum(amps)/len(amps):.3f}, "
                        f"min={min(amps):.3f}, "
                        f"max={max(amps):.3f}"
                    )
    
        return spike_times, spike_amplitudes

            
            
    def reset(self):
        """
        Reset the NEURON simulation state and internal recording buffers.
        Clears stored voltage traces, stimuli, and resets the simulation clock.

        Returns
        -------
        None
        """
        h.finitialize(-70)
        h.t = 0
        self.t_vec = None
        self.v_records = {}
        self.stimulas = []
        return

        