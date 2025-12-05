import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Union, Set
import os
import hashlib
import json
import itertools
import glob # Для поиска файлов

# --- Типы данных ---
DirectednessMap = Dict[str, Dict[str, bool]] 
InitialValueMap = Dict[str, Union[float, int]]

# ==============================================================================
# 1. JaxGraphConverter
# ==============================================================================

class JaxGraphConverter:
    """
    Преобразует граф NetworkX с произвольными ID и типами узлов 
    в наборы массивов ребер и карт маппинга, готовых для использования в JAX GNN.
    """

    def __init__(self,
                 graph: Union[nx.DiGraph, nx.MultiDiGraph],
                 node_type_groups: Dict[str, List[str]],
                 edge_directedness: DirectednessMap):
        
        self.graph = graph
        self.node_type_groups = node_type_groups
        self.edge_directedness = edge_directedness
        
        self.num_nodes: Dict[str, int] = {}             
        self.edge_arrays: Dict[str, np.ndarray] = {}    
        self.global_mapping: Dict[Any, int] = {}
        self.local_maps: Dict[str, Dict[int, int]] = {} 

        self._process_graph() # Вызов, который приводил к ошибке

    @staticmethod
    def load_from_npz(filepath: str) -> Dict[str, Any]:
        """
        Загружает результаты конвертации (включая маппинги) из NPZ-файла.
        """
        data = np.load(filepath, allow_pickle=True)
        results: Dict[str, Any] = {}
        num_nodes: Dict[str, int] = {}
        
        for key in data.files:
            if key.startswith('num_'):
                # Извлечение скалярного значения для количества узлов
                num_nodes[key.split('_')[1]] = data[key].item() 
            elif key.startswith('edges_'):
                results[key] = data[key]
            elif key == 'local_maps':
                # Извлечение словаря с локальными маппингами
                results['local_maps'] = data[key].item() 
            elif key == 'global_mapping':
                # Извлечение словаря с глобальным маппингом
                results['mapping'] = {'old_to_new_global': data[key].item()}
        
        results['num_nodes'] = num_nodes
        
        print(f"   Загружено: {len(results['num_nodes'])} групп, {len([k for k in results if k.startswith('edges_')])} массивов ребер.")
        return results

    def save_to_npz(self, filepath: str):
        """
        Сохраняет массивы ребер, количество узлов и карты маппинга в NPZ-файл.
        """
        data_to_save: Dict[str, Any] = {}
        
        # 1. Массивы ребер
        data_to_save.update(self.edge_arrays)
        
        # 2. Количество узлов
        for group, count in self.num_nodes.items():
            data_to_save[f'num_{group}'] = np.array(count)
            
        # 3. Карты маппинга (сохраняем как объекты)
        data_to_save['local_maps'] = np.array(self.local_maps, dtype=object)
        data_to_save['global_mapping'] = np.array(self.global_mapping, dtype=object)
        
        np.savez_compressed(filepath, **data_to_save)
        print(f"   Данные графа успешно сохранены в {filepath} (ключи: {list(data_to_save.keys())})")

    def get_results(self) -> Dict[str, Any]:
        """
        Возвращает результаты конвертации, включая маппинги.
        """
        return {
            'num_nodes': self.num_nodes,
            'mapping': {'old_to_new_global': self.global_mapping},
            'local_maps': self.local_maps,
            **self.edge_arrays
        }

    def _process_graph(self):
        """Оркестрация процесса конвертации: создание маппингов и массивов ребер.
        Этот метод был пропущен, что вызвало AttributeError."""
        self._create_mappings()
        self._create_edge_arrays()

    def _get_group_name(self, node_id: Any) -> List[str]:
        """
        Определяет группы узла по его атрибуту 'type'.
        Возвращает список имен групп, к которым принадлежит узел.
        """
        node_type = self.graph.nodes[node_id].get('type')
        if not node_type:
            return [] # Возвращаем пустой список, если нет типа
        
        found_groups: List[str] = []
        for group_name, type_list in self.node_type_groups.items():
            if node_type in type_list:
                found_groups.append(group_name)
        return found_groups

    def _create_mappings(self):
        """Создает карты маппинга старых ID в новые глобальные и локальные индексы."""
        
        sorted_nodes = sorted(self.graph.nodes())
        
        global_idx = 0
        for node_id in sorted_nodes:
            self.global_mapping[node_id] = global_idx
            global_idx += 1
            
        self.local_maps = {group: {} for group in self.node_type_groups.keys()}
        self.num_nodes = {group: 0 for group in self.node_type_groups.keys()}

        for node_id in sorted_nodes:
            # Использование новой функции, которая возвращает список групп
            group_names = self._get_group_name(node_id) 
            
            for group_name in group_names: # Цикл по всем группам, к которым принадлежит узел
                if group_name in self.local_maps:
                    global_id = self.global_mapping[node_id]
                    
                    # Проверяем, не был ли узел уже добавлен в маппинг этой группы
                    # (Это не должно случиться, если узел принадлежит только одной группе через type_list,
                    # но важно для устойчивости, если логика node_type_groups изменится)
                    if global_id not in self.local_maps[group_name]:
                        local_idx = self.num_nodes[group_name]
                        
                        self.local_maps[group_name][global_id] = local_idx
                        self.num_nodes[group_name] += 1
                
        self.num_nodes = {k: v for k, v in self.num_nodes.items() if v > 0}
        self.local_maps = {k: v for k, v in self.local_maps.items() if v}


    def _create_edge_arrays(self):
        """
        Создает массивы ребер (src, dst) с использованием локальных индексов 
        для всех комбинаций групп.
        """
        groups = list(self.num_nodes.keys())
        
        for src_group, dst_group in itertools.product(groups, groups):
            edge_list: List[Tuple[int, int]] = []
            is_directed = self.edge_directedness.get(src_group, {}).get(dst_group, True)
            
            src_local_map = self.local_maps.get(src_group, {})
            dst_local_map = self.local_maps.get(dst_group, {})

            if not src_local_map or not dst_local_map:
                continue 

            for u_old, v_old in self.graph.edges():
                # Использование новой функции, которая возвращает список групп
                u_groups = self._get_group_name(u_old)
                v_groups = self._get_group_name(v_old)
                
                # Проверяем, что исходный узел принадлежит src_group, а конечный - dst_group
                # Так как узел может принадлежать нескольким группам, используем циклы
                for u_group in u_groups:
                    for v_group in v_groups:
                        if u_group == src_group and v_group == dst_group:
                            u_global_id = self.global_mapping[u_old]
                            v_global_id = self.global_mapping[v_old]
                            
                            u_local_idx = src_local_map.get(u_global_id)
                            v_local_idx = dst_local_map.get(v_global_id)
                            
                            if u_local_idx is not None and v_local_idx is not None:
                                edge_list.append((u_local_idx, v_local_idx))
                                # Учитывая, что мы добавляем ребро только один раз
                                # для конкретной пары (src_group, dst_group), 
                                # можно прервать внутренние циклы после первого совпадения.
                                break 
                    if u_group == src_group:
                        break # Прерываем внешний цикл, если нашли соответствие src_group

            if not is_directed:
                # Логика для создания симметричных ребер
                if src_group == dst_group:
                    undirected_edges = set(edge_list)
                    for u, v in edge_list:
                        undirected_edges.add((v, u))
                    edge_list = list(undirected_edges)
                
            
            if edge_list:
                # Удаление дубликатов на всякий случай, если граф был MultiDiGraph
                # и мы добавили ребро несколько раз
                unique_edges = sorted(list(set(edge_list)))
                array = np.array(unique_edges, dtype=np.int32).T 
                key = f'edges_{src_group}_to_{dst_group}'
                self.edge_arrays[key] = array
                

# ==============================================================================
# 2. SimulationContextJax (Исправленный с новым статическим методом)
# ==============================================================================

class SimulationContextJax:
    
    CACHE_FILENAME = "graph_arrays.npz"
    HASH_FILENAME = "graph_hash.txt"

    def __init__(self,
                 graph: Union[nx.DiGraph, nx.MultiDiGraph],
                 node_type_groups: Dict[str, List[str]],
                 edge_directedness: DirectednessMap,
                 initial_node_values: InitialValueMap,
                 cache_dir: Union[str, None] = None):
        
        self.node_type_groups = node_type_groups
        self.edge_directedness = edge_directedness
        self.initial_node_values = initial_node_values
        self.cache_dir = cache_dir
        self.graph_results: Dict[str, Any] = {}
        self.initial_states: Dict[str, np.ndarray] = {}
        
        # 1. Запуск логики кэширования
        cache_hit = False
        if self.cache_dir:
            # Для хэша нужно использовать обновленную логику определения групп, 
            # но поскольку JaxGraphConverter не используется тут напрямую, 
            # оставим хэширование на основе исходной конфигурации.
            self.graph_hash = self._calculate_graph_hash(graph, node_type_groups, edge_directedness)
            cache_hit = self._try_load_from_cache()

        if cache_hit:
            print("1. Данные графа успешно загружены из кэша.")
        else:
            print("1. Кэш пропущен/не найден. Запуск JaxGraphConverter для обработки графа...")
            converter = JaxGraphConverter(graph, node_type_groups, edge_directedness)
            self.graph_results = converter.get_results()
            
            if self.cache_dir:
                self._save_to_cache(converter)
        
        # Извлечение необходимых маппингов и размеров
        self.num_nodes = self.graph_results['num_nodes']
        self.local_maps = self.graph_results['local_maps']
        
        print("2. Инициализация начальных массивов состояний узлов...")
        self._initialize_node_states()

        print("Контекст симуляции JAX готов.")

    # --------------------------------------------------------------------------
    # СТАТИЧЕСКИЙ МЕТОД: Загрузка контекста из кэша
    # --------------------------------------------------------------------------
    @staticmethod
    def load_context_from_cache(cache_dir: str, 
                                initial_node_values: InitialValueMap) -> Dict[str, Any]:
        """
        Загружает данные графа и генерирует начальные состояния узлов 
        на основе кэшированных массивов. Включает маппинг в возвращаемый контекст.

        Args:
            cache_dir (str): Директория, где находятся подпапки кэша (./jax_context_cache).
            initial_node_values (InitialValueMap): Конфигурация для создания 
                                                   начальных состояний.

        Returns:
            Dict[str, Any]: Словарь контекста, готовый для GNN модели, включая 'mapping'.
        """
        if not os.path.exists(cache_dir):
            raise FileNotFoundError(f"Директория кэша не найдена: {cache_dir}")

        # 1. Найти последнюю поддиректорию кэша (по имени хэша)
        all_subdirs = [d for d in glob.glob(os.path.join(cache_dir, '*')) 
                       if os.path.isdir(d)]
        
        if not all_subdirs:
            raise FileNotFoundError(f"Не найдены поддиректории кэша в: {cache_dir}")

        latest_subdir = max(all_subdirs, key=os.path.getmtime)
        
        npz_path = os.path.join(latest_subdir, SimulationContextJax.CACHE_FILENAME)
        
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Файл кэша NPZ не найден: {npz_path}")

        print(f"Загрузка контекста из: {npz_path}")
        
        # 2. Загрузка данных графа из NPZ
        graph_data = JaxGraphConverter.load_from_npz(npz_path)

        num_nodes = graph_data['num_nodes']
        local_maps_global_to_local = graph_data.get('local_maps', {})
        global_map_old_to_new = graph_data.get('mapping', {}).get('old_to_new_global', {})

        edge_arrays = {k: v for k, v in graph_data.items() if k.startswith('edges_')}
        initial_states: Dict[str, np.ndarray] = {}
        
        # 3. Инициализация начальных состояний 
        for group_name, num_n in num_nodes.items():
            
            if group_name not in initial_node_values:
                continue

            value = initial_node_values[group_name]
            
            if isinstance(value, (int, float)) and value != 0: 
                if value <= 1:
                    initial_value = float(value)
                    shape = (num_n,)
                    initial_states[group_name] = np.full(shape, initial_value, dtype=np.float32)
                elif isinstance(value, int) and value > 1:
                    dimension = value
                    shape = (num_n, dimension)
                    initial_states[group_name] = np.zeros(shape, dtype=np.float32)
        
        # 4. Расчет конечного маппинга (логика скопирована из get_node_id_mapping)
        # Эта логика остается прежней, так как она работает с уже загруженными структурами
        final_mapping: Dict[str, Dict[Any, int]] = {k: {} for k in num_nodes.keys()}
        global_to_group: Dict[int, str] = {}
        
        for group_name, map_dict in local_maps_global_to_local.items():
            # Здесь map_dict - это Dict[int, int] (global_id -> local_id)
            for global_id, local_id in map_dict.items():
                # Так как узел может принадлежать нескольким группам, 
                # global_id может быть переписан. В данном контексте это 
                # не проблема, так как нам нужна любая группа, к которой он принадлежит, 
                # чтобы найти его оригинальный ID, но для точности лучше использовать
                # все группы, к которым он принадлежит.
                # Однако, для восстановления финального маппинга,
                # нам просто нужно сопоставить old_id -> local_id.
                
                # Если узел принадлежит нескольким группам, он будет иметь несколько 
                # локальных ID (по одному для каждой группы). 
                # Глобальный ID уникален, а local_maps_global_to_local хранит 
                # его локальный ID в каждой группе.
                pass # Не требуется создавать global_to_group таким способом, 
                     # так как global_map_old_to_new уже содержит всю информацию
                     
        # Создание final_mapping (old_id -> local_id)
        # Используем local_maps_global_to_local, чтобы найти local_id для каждого old_id
        for original_id, global_id in global_map_old_to_new.items():
            # Перебираем все группы, чтобы найти local_id для этого global_id
            for group_name in num_nodes.keys():
                local_map = local_maps_global_to_local.get(group_name, {})
                local_index = local_map.get(global_id)
                
                if local_index is not None:
                    if group_name not in final_mapping:
                        final_mapping[group_name] = {}
                    final_mapping[group_name][original_id] = local_index
        
        # 5. Сборка конечного контекста
        context = {
            'num_nodes': num_nodes,
            'initial_states': initial_states,
            'mapping': final_mapping, # <<< ДОБАВЛЕНО: Конечный маппинг
            **edge_arrays
        }
        
        print("Контекст симуляции успешно собран из кэша.")
        return context

    # --------------------------------------------------------------------------
    # Методы кэширования и инициализации 
    # --------------------------------------------------------------------------

    @staticmethod
    def _calculate_graph_hash(graph: Union[nx.DiGraph, nx.MultiDiGraph], node_type_groups: Dict, edge_directedness: Dict) -> str:
        """Генерирует SHA256 хэш."""
        # Для корректного хэширования в многогрупповом режиме:
        # 1. Сортируем узлы по ID.
        # 2. Для каждого узла включаем его ID и его тип (для учета группы).
        # 3. Включаем полную конфигурацию групп и направленности.
        node_data = sorted([(str(n), graph.nodes[n].get('type')) for n in graph.nodes()])
        node_str = json.dumps(node_data, sort_keys=True)
        edge_data = sorted([(str(u), str(v)) for u, v in graph.edges()])
        edge_str = json.dumps(edge_data, sort_keys=True)
        config_data = {"groups": node_type_groups, "directedness": edge_directedness}
        config_str = json.dumps(config_data, sort_keys=True)
        full_string = (node_str + edge_str + config_str).encode('utf-8')
        return hashlib.sha256(full_string).hexdigest()

    def _get_cache_paths(self) -> Tuple[str, str]:
        """Возвращает полные пути к файлам NPZ и HASH."""
        if not self.cache_dir:
            raise ValueError("Cache directory is not set.")
        cache_sub_dir = os.path.join(self.cache_dir, self.graph_hash)
        os.makedirs(cache_sub_dir, exist_ok=True)
        npz_path = os.path.join(cache_sub_dir, self.CACHE_FILENAME)
        hash_path = os.path.join(cache_sub_dir, self.HASH_FILENAME)
        return npz_path, hash_path

    def _try_load_from_cache(self) -> bool:
        """Попытка загрузить данные графа из кэша. Возвращает True при успехе."""
        try:
            npz_path, hash_path = self._get_cache_paths()
        except ValueError:
            return False

        if not os.path.exists(npz_path) or not os.path.exists(hash_path):
            return False
            
        try:
            with open(hash_path, 'r') as f:
                saved_hash = f.read().strip()
                
            if saved_hash == self.graph_hash:
                self.graph_results = JaxGraphConverter.load_from_npz(npz_path)
                return True
            else:
                return False
                
        except Exception as e:
            return False

    def _save_to_cache(self, converter: JaxGraphConverter):
        """Сохраняет сгенерированные массивы и хэш в кэш директорию."""
        try:
            npz_path, hash_path = self._get_cache_paths()
            converter.save_to_npz(npz_path)
            with open(hash_path, 'w') as f:
                f.write(self.graph_hash)
            print(f"   Данные графа и хэш успешно сохранены в кэш: {os.path.dirname(npz_path)}")
        except Exception as e:
            print(f"Внимание: Не удалось сохранить данные в кэш: {e}")

    def _initialize_node_states(self):
        """
        Создает массивы NumPy для начальных состояний узлов.
        """
        for group_name, num_n in self.num_nodes.items():
            if group_name not in self.initial_node_values:
                continue

            value = self.initial_node_values[group_name]
            
            if isinstance(value, (int, float)) and value != 0: 
                if value <= 1:
                    initial_value = float(value)
                    shape = (num_n,)
                    self.initial_states[group_name] = np.full(shape, initial_value, dtype=np.float32)
                elif isinstance(value, int) and value > 1:
                    dimension = value
                    shape = (num_n, dimension)
                    self.initial_states[group_name] = np.zeros(shape, dtype=np.float32)
    
    
    def get_node_id_mapping(self) -> Dict[str, Dict[Any, int]]:
        """
        Возвращает маппинг старых (оригинальных) ID узлов в новые (локальные) индексы
        для каждой группы узлов.
        """
        global_map = self.graph_results.get('mapping', {}).get('old_to_new_global', {})
        local_maps_global_to_local = self.local_maps
        
        final_mapping: Dict[str, Dict[Any, int]] = {k: {} for k in self.num_nodes.keys()}
        
        # Создание final_mapping (old_id -> local_id)
        for original_id, global_id in global_map.items():
            # Перебираем все группы, чтобы найти local_id для этого global_id
            for group_name in self.num_nodes.keys():
                local_map = local_maps_global_to_local.get(group_name, {})
                local_index = local_map.get(global_id)
                
                if local_index is not None:
                    # Узел принадлежит этой группе, добавляем его в маппинг
                    final_mapping[group_name][original_id] = local_index
            
        return final_mapping

    def get_context(self) -> Dict[str, Any]:
        """Возвращает полный контекст для симуляции GNN, включая маппинг."""
        # Собираем массивы ребер из graph_results
        edge_arrays = {k: v for k, v in self.graph_results.items() if k.startswith('edges_')}

        return {
            'num_nodes': self.num_nodes,
            'initial_states': self.initial_states,
            'mapping': self.get_node_id_mapping(), # <<< ДОБАВЛЕНО: Конечный маппинг
            **edge_arrays
        }


# --- Демонстрация использования ---
if __name__ == '__main__':
    
    # ВРЕМЕННЫЙ КЭШ ДИРЕКТОРИЯ
    CACHE_DIR = './jax_context_cache' 
    import shutil

    if os.path.exists(CACHE_DIR):
        # Удаляем старую директорию, чтобы гарантировать "cache miss" для Теста 1
        shutil.rmtree(CACHE_DIR)
        print(f"Удалена старая тестовая директория кэша '{CACHE_DIR}'.")
    
    # 1. Создание примера графа NetworkX и конфигурация
    G = nx.DiGraph() 
    # ВНИМАНИЕ: Для теста новой логики узел 1001 будет принадлежать только 'H', 
    # но можно было бы добавить тип, который принадлежит нескольким группам, 
    # чтобы проверить новую логику. В данном примере типы H_root и S_conn 
    # принадлежат только одной группе (H или S). 
    # Давайте добавим новый тип, который принадлежит обеим группам для теста.
    
    # Модифицированная конфигурация групп для теста:
    # Пусть тип 'HS_hybrid' принадлежит и 'H', и 'S'.
    type_groups = {'H': ['H_root', 'HS_hybrid'], 'S': ['S_conn', 'HS_hybrid']}
    
    node_configs = {
        1001: 'H_root', 1005: 'H_root', 
        2010: 'S_conn', 2015: 'H_root', 
        1020: 'S_conn', 1025: 'H_root',
        2030: 'H_root', 
        2035: 'S_conn',
        # Узел-гибрид
        9000: 'HS_hybrid' 
    }
    for oid, ntype in node_configs.items():
        G.add_node(oid, type=ntype)
    
    G.add_edges_from([
        (1001, 1005), (1005, 1001),         # H_root -> H_root
        (9000, 1001), (1001, 9000),         # HS_hybrid -> H_root (H->H, S->H)
        (2010, 9000), (9000, 2010),         # S_conn -> HS_hybrid (S->S, S->H, H->S, H->S)
        (2010, 1001),                       # S_conn -> H_root
        (1005, 1020),                       # H_root -> S_conn
        (2035, 2010),                       # S_conn -> S_conn
        (1025, 2030)                        # H_root -> H_root
    ])
    
    # type_groups = {'H': ['H_root', 'HS_hybrid'], 'S': ['S_conn', 'HS_hybrid']}        
    directedness = {'H': {'H': False, 'S': True}, 'S': {'H': True, 'S': True}}
    initial_values_config = {'H': 0.5, 'S': 3} # 0.5 для H, размерность 3 для S
    
    
    # 2. ПЕРВЫЙ ЗАПУСК: Создание и сохранение кэша
    print("\n================== ТЕСТ 1: СОЗДАНИЕ КЭША (с гибридным узлом 9000) ==================")
    context_manager = SimulationContextJax(
        graph=G,
        node_type_groups=type_groups,
        edge_directedness=directedness,
        initial_node_values=initial_values_config,
        cache_dir=CACHE_DIR 
    )
    
    # Проверка, что context_manager создал кэш
    first_context = context_manager.get_context()
    
    print("\nРазмеры массивов в первом контексте:")
    print(f"  > H-узлов: {first_context['num_nodes']['H']} (Ожидается: 6, включая 9000)")
    print(f"  > S-узлов: {first_context['num_nodes']['S']} (Ожидается: 4, включая 9000)")
    print(f"  > H_to_H ребер: {first_context.get('edges_H_to_H', 'N/A').shape if isinstance(first_context.get('edges_H_to_H'), np.ndarray) else 'N/A'}")
    
    h_map = first_context['mapping']['H']
    s_map = first_context['mapping']['S']
    
    print(f"  > Узел 9000 в маппинге H: {'Да' if 9000 in h_map else 'Нет'} (Локальный индекс: {h_map.get(9000)})")
    print(f"  > Узел 9000 в маппинге S: {'Да' if 9000 in s_map else 'Нет'} (Локальный индекс: {s_map.get(9000)})")


    # 3. ВТОРОЙ ЗАПУСК: Загрузка контекста без пересчета графа
    print("\n================== ТЕСТ 2: ЗАГРУЗКА ИЗ КЭША ==================")
    del context_manager # Удаляем старый объект
    
    try:
        loaded_context = SimulationContextJax.load_context_from_cache(
            cache_dir=CACHE_DIR,
            initial_node_values=initial_values_config
        )
        
        print("\nРазмеры массивов в загруженном контексте:")
        print(f"  > H-узлов: {loaded_context['num_nodes']['H']}")
        print(f"  > S-узлов: {loaded_context['num_nodes']['S']}")
        print(f"  > H_to_H ребер: {loaded_context.get('edges_H_to_H', 'N/A').shape if isinstance(loaded_context.get('edges_H_to_H'), np.ndarray) else 'N/A'}")

        h_map_loaded = loaded_context['mapping']['H']
        s_map_loaded = loaded_context['mapping']['S']

        print(f"  > Узел 9000 в маппинге H (загр.): {'Да' if 9000 in h_map_loaded else 'Нет'} (Локальный индекс: {h_map_loaded.get(9000)})")
        print(f"  > Узел 9000 в маппинге S (загр.): {'Да' if 9000 in s_map_loaded else 'Нет'} (Локальный индекс: {s_map_loaded.get(9000)})")


        # Проверка соответствия (для демонстрации)
        assert loaded_context['num_nodes']['H'] == 6
        assert loaded_context['num_nodes']['S'] == 4
        assert loaded_context['mapping'] == first_context['mapping']
        print("\nПроверка: Контекст успешно загружен и совпадает с оригинальным (включая маппинг гибридного узла).")

    except Exception as e:
        print(f"ОШИБКА ПРИ ЗАГРУЗКЕ КОНТЕКСТА ИЗ КЭША: {e}")

    # --- Очистка кэша ---
    try:
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
    except Exception as e:
        pass