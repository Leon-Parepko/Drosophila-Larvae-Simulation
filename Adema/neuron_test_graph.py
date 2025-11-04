import os
import numpy as np
import pandas as pd
import networkx as nx
from collections import deque
from neuron import h

import matplotlib
matplotlib.use('Agg')  # causes some error with visualization
import matplotlib.pyplot as plt

h.load_file('stdrun.hoc')

NEURONS_DIR = 'Neurons Metadata/Graphs/neurons'  # GML files path for eahc neuron
META_PKL = 'Neurons Metadata/Metadata/Metadata(auto).pkl'
NEURONS = ['7594047', '15741865', '18833414']  # какие нейроны загружать


def set_segments(section):
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


class Network:
    def __init__(self):
        self.graphs = {}
        self.sections = {}
        self.somas = {}

        self.synapses = []
        self.t_vec = None
        self.v_records = {}

        print('number of neurons:', len(NEURONS), '\n')

    def load_graphs(self):
        print('loading graphs \n')

        for neuron_id in NEURONS:
            path = os.path.join(NEURONS_DIR, f'{neuron_id}.gml')

            if not os.path.exists(path):
                raise FileNotFoundError(f'{path} not found')

            # read graph
            graph = nx.read_gml(path)
            self.graphs[neuron_id] = graph

            # root - soma
            root_nodes = [n for n, data in graph.nodes(data=True)
                          if data.get('type') == 'root']

            print(f'{neuron_id}-{graph.number_of_nodes()} nodes, root={root_nodes[0] if root_nodes else 'NONE'}')

    def build_sections(self):
        print('building sections \n')

        for neuron_id, graph in self.graphs.items():

            for node_id, data in graph.nodes(data=True):
                # if soma
                if data.get('type') == 'root':
                    # soma - hh
                    soma = h.Section(name=f"soma_{neuron_id}")

                    soma.L = 20 # длина
                    soma.diam = 20 # диаметр
                    soma.Ra = 100 # аксиальное сопротивление
                    soma.cm = 1 # емкость мембраны

                    soma.insert('hh')

                    soma.gnabar_hh = 0.12 # проводимость натриевых каналов
                    soma.gkbar_hh = 0.036 # проводимость калиевых каналов
                    soma.gl_hh = 0.0003 # leak проводимость
                    soma.el_hh = -54.3

                    # set number of segments
                    set_segments(soma)

                    self.sections[(neuron_id, node_id)] = soma
                    self.somas[neuron_id] = soma
                    break

            # create dendrites
            dendrite_count = 0

            for node_id, data in graph.nodes(data=True):
                node_type = data.get('type')

                if node_type in ('slab', 'branch', 'end'):
                    dend = h.Section(name=f"dend_{neuron_id}_{node_id}")

                    dend.L = 50.0  # длина
                    dend.diam = 2.0  # диаметр
                    dend.Ra = 100.0  # сопротивление
                    dend.cm = 1.0  # емкость

                    # dendrites - passive
                    dend.insert('pas')  # passive механизм
                    dend.g_pas = 0.0001  # пассивная проводимость
                    dend.e_pas = -70.0  # reversal potential

                    set_segments(dend)

                    self.sections[(neuron_id, node_id)] = dend
                    dendrite_count += 1

            print(f'{neuron_id}: {dendrite_count} dendrites')

    def connect_morphology(self):
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

            print(f'{neuron_id}: {connections} connections')

    def build_synapses(self):
        print('building synapses \n')

        meta = pd.read_pickle(META_PKL)
        meta = meta.set_index('skeleton_id')
        meta.index = meta.index.astype(str)

        connector_tables = {}

        for neuron_id in NEURONS:
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
            print(f'{neuron_id}: {len(df)} connectors')

        synapse_count = 0

        for i, neuron_a in enumerate(NEURONS):
            for neuron_b in NEURONS[i + 1:]:

                if neuron_a not in connector_tables or neuron_b not in connector_tables:
                    continue

                # merge таблицы по connector_id
                merged = connector_tables[neuron_a].merge(connector_tables[neuron_b], on='connector_id', suffixes=('_a', '_b'))

                # если нет общих коннекторов - пропускаем
                if len(merged) == 0:
                    continue

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
                        post_neuron = neuron_b  # сюда приходит
                        post_node = node_b  # узел где синапс

                    # если A=1 и B=0, то B -> A
                    elif bit_a == 1 and bit_b == 0:
                        pre_neuron = neuron_b
                        post_neuron = neuron_a
                        post_node = node_a

                    # если оба 0 или оба 1 - пропускаем
                    else:
                        continue

                    post_section = self.sections.get((post_neuron, post_node))
                    if not post_section:
                        continue

                    pre_soma = self.somas.get(pre_neuron)
                    if not pre_soma:
                        continue

                    # создаем синапс на постсинаптическом дендрите
                    synapse = h.Exp2Syn(post_section(0.5))  # в центре секции

                    synapse.tau1 = 0.5  # время нарастания
                    synapse.tau2 = 3.0  # время спада
                    synapse.e = 0.0  # reversal potential

                    # создаем детектор спайков - voltage сомы пресинаптика
                    netcon = h.NetCon(
                        pre_soma(0.5)._ref_v,  # откуда берем voltage
                        synapse,  # куда отправляем event
                        sec=pre_soma
                    )

                    # Параметры NetCon
                    netcon.threshold = 0.0  # порог (мВ) - спайк когда V > 0
                    netcon.weight[0] = 0.5  # вес синапса (микросименсы)
                    netcon.delay = 3.0  # задержка (ms)

                    # Сохраняем
                    self.synapses.append(netcon)
                    synapse_count += 1

        print(f'Synapses created: {synapse_count}')

    def setup_recording(self):
        print('setting recordings \n')

        # вектор для записи времени
        self.t_vec = h.Vector().record(h._ref_t)

        for neuron_id, soma in self.somas.items():
            # создаем Vector и записываем voltage из центра сомы
            vec = h.Vector().record(soma(0.5)._ref_v)

            self.v_records[neuron_id] = vec

    def setup_stimulus(self):
        print('adding stimulus \n')

        # take the first neuron
        first_neuron = NEURONS[0]

        soma = self.somas[first_neuron]

        # создаем электрод
        stim = h.IClamp(soma(0.5))  # в центре сомы

        # Параметры стимуляции
        stim.delay = 10.0  # время начало
        stim.dur = 5.0  # длительность
        stim.amp = 2.0  # сила тока

        self.stimulus = stim

        print(f'Stimulating {first_neuron}: {stim.amp}for {stim.dur}')

    def run(self, duration=100):

        print(f'running simulation for {duration}\n')

        h.dt = 0.025  # шаг времени
        h.steps_per_ms = 40  # количество шагов на миллисекунду
        h.v_init = -70  # начальный voltage

        h.finitialize(-70)

        # Run simulation
        h.continuerun(duration)

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
                print(f'{neuron_id}: SPIKE at {spike_time}')
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


def plot_results(t, voltages):
    plt.figure(figsize=(12, 6))
    colors = ['red', 'blue', 'green']

    for i, (neuron_id, v) in enumerate(voltages.items()):
        plt.plot(t, v, label=neuron_id, color=colors[i], linewidth=2)

    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Network Activity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvline(x=10, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    # создаем сеть
    net = Network()

    # строим сеть
    net.load_graphs()
    net.build_sections()
    net.connect_morphology()
    net.build_synapses()
    net.setup_recording()
    net.setup_stimulus()

    # запускаем симуляцию
    t, voltages = net.run(duration=100)

    net.analyze(t, voltages)
    plot_results(t, voltages)