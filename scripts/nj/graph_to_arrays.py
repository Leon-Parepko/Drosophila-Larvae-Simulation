import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any, Union
import os

# Словарь направленности: Dict[GroupU][GroupV] -> bool (True=Directed, False=Undirected)
DirectednessMap = Dict[str, Dict[str, bool]]

class JaxGraphConverter:
    """
    Класс для переиндексации графа NetworkX и подготовки массивов ребер 
    для вычислительных бэкендов, таких как JAX/Haiku/GNNs.

    Он сопоставляет исходные ID узлов новым, последовательным глобальным индексам, 
    разделяет узлы на определенные группы (например, 'H' и 'S') и генерирует 
    массивы ребер, используя локальные индексы внутри этих групп.
    """

    def __init__(self, 
                 graph: Union[nx.DiGraph, nx.MultiDiGraph], 
                 node_type_groups: Dict[str, List[str]],
                 edge_directedness: DirectednessMap):
        """
        Инициализирует конвертер и запускает процесс преобразования.

        Args:
            graph: Граф NetworkX (DiGraph или MultiDiGraph) для обработки. 
                   Узлы должны иметь атрибут 'type'.
            node_type_groups: Словарь, сопоставляющий абстрактные имена групп (например, 'H', 'S') 
                              со списком значений атрибута 'type' узлов графа.
                              Пример: {'H': ['root', 'branch'], 'S': ['connector']}
            edge_directedness: Словарь, определяющий направленность связей между группами.
                               Для ненаправленных межгрупповых связей (U != V), достаточно
                               указать направленность для канонической пары (min_group, max_group).
        """
        # Преобразуем MultiDiGraph в DiGraph, игнорируя ключи ребер, так как для JAX нужны только u->v
        self.graph = nx.DiGraph(graph) if isinstance(graph, nx.MultiDiGraph) else graph
            
        self.node_type_groups = node_type_groups
        self.edge_directedness = edge_directedness
        self.group_names = sorted(list(node_type_groups.keys())) # Сортируем для детерминированности
        
        # Внутреннее состояние для заполнения
        self.local_maps: Dict[str, Dict[int, int]] = {name: {} for name in self.group_names}
        self.group_nodes: Dict[str, List[int]] = {name: [] for name in self.group_names}
        self.node_type_map: Dict[int, str] = {} # global_id -> group_name

        self.results: Dict[str, Any] = {}

        self._reindex_and_map_nodes()
        self.convert_to_jax_arrays()

    def _get_group_name(self, node_data: Dict[str, Any]) -> str:
        """Определяет абстрактное имя группы ('H', 'S', и т.д.) для данного узла."""
        node_type = str(node_data.get('type')) # Приводим к строке на всякий случай
        if node_type is None:
             raise ValueError(f"У узла отсутствует атрибут 'type', необходимый для группировки.")
             
        for group_name, types in self.node_type_groups.items():
            if node_type in types:
                return group_name
        
        raise ValueError(f"Тип узла '{node_type}' не найден ни в одной из определенных групп: {self.node_type_groups.keys()}")

    def _reindex_and_map_nodes(self):
        """
        1. Создает глобальную переиндексацию (old_id -> new_global_id).
        2. Разделяет узлы на группы и назначает локальные индексы (new_global_id -> new_local_id).
        """
        
        # 1. Глобальная переиндексация (0 до N-1)
        original_ids = [str(node) for node in self.graph.nodes()]
        old_to_new_global = {old_id: new_id for new_id, old_id in enumerate(original_ids)}
        new_to_old_global = {v: k for k, v in old_to_new_global.items()}
        
        # Для NetworkX relabel_nodes нужна исходная мапа с исходными типами
        original_map = {node: old_to_new_global[str(node)] for node in self.graph.nodes()}
        reindexed_graph = nx.relabel_nodes(self.graph, original_map, copy=True)
        self.results['reindexed_graph'] = reindexed_graph
        
        # Сохраняем маппинг
        self.results['mapping'] = {
            "old_to_new_global": old_to_new_global,
            "new_to_old_global": new_to_old_global,
        }

        # 2. Локальная переиндексация и маппинг типов
        sorted_nodes = sorted(reindexed_graph.nodes(data=True), key=lambda x: x[0])

        for global_id, data in sorted_nodes:
            try:
                group_name = self._get_group_name(data)
            except ValueError as e:
                print(f"Ошибка группировки узла {global_id}: {e}")
                continue
            
            self.node_type_map[global_id] = group_name
            
            current_local_index = len(self.group_nodes[group_name])
            self.local_maps[group_name][global_id] = current_local_index
            self.group_nodes[group_name].append(global_id)
            
        # Сохраняем количество узлов
        self.results['num_nodes'] = {name: len(nodes) for name, nodes in self.group_nodes.items()}
        print(f"Узлы успешно сгруппированы: {self.results['num_nodes']}")

    def _is_directed(self, u_group: str, v_group: str) -> bool:
        """ 
        Проверяет, является ли связь между u_group и v_group направленной.
        По умолчанию, если не указано, связь считается направленной (True).
        Для ненаправленных межгрупповых связей (U != V), проверяется канонический ключ.
        """
        
        if u_group == v_group:
            # Самопетли: проверка только по u_group -> u_group
            return self.edge_directedness.get(u_group, {}).get(u_group, True)
        
        # Для связей U != V: используем канонический ключ группы для проверки
        key_u, key_v = tuple(sorted((u_group, v_group)))
        
        # Проверяем по каноническому ключу key_u -> key_v
        return self.edge_directedness.get(key_u, {}).get(key_v, True)


    def convert_to_jax_arrays(self):
        """
        Генерирует массивы ребер (u_local, v_local) для всех комбинаций групп, 
        учитывая флаг направленности.
        """
        
        reindexed_graph = self.results['reindexed_graph']
        
        # Инициализация вложенной структуры списков ребер: edge_lists[U_GROUP][V_GROUP]
        edge_lists: Dict[str, Dict[str, List[Tuple[int, int]]]] = {u_group: {} for u_group in self.group_names}
        for u_group in self.group_names:
            for v_group in self.group_names:
                edge_lists[u_group][v_group] = []
                
        # 1. Сбор всех локальных ребер
        for u_global, v_global in reindexed_graph.edges():
            u_group = self.node_type_map.get(u_global)
            v_group = self.node_type_map.get(v_global)
            
            if u_group is None or v_group is None:
                continue

            u_local = self.local_maps[u_group][u_global]
            v_local = self.local_maps[v_group][v_global]

            edge_lists[u_group][v_group].append((u_local, v_local))
            
        # 2. Обработка, дедупликация и сохранение массивов NumPy
        num_logical_edge_combinations = 0
        
        # Множество для отслеживания обратных ненаправленных пар, которые уже были обработаны
        processed_undirected_pairs = set()

        for u_group in self.group_names:
            for v_group in self.group_names:
                
                key = f"edges_{u_group}_to_{v_group}"
                edges = edge_lists[u_group][v_group]
                
                # Пропускаем, если это обратное направление ненаправленной связи, которое уже обработано
                if (u_group, v_group) in processed_undirected_pairs:
                    continue
                
                directed = self._is_directed(u_group, v_group)
                final_edges = []
                
                if directed:
                    # Case 1: Направленная связь (включает U->V и U->U по умолчанию)
                    final_edges = edges
                    num_logical_edge_combinations += 1
                    
                elif u_group == v_group:
                    # Case 2: Ненаправленные самопетли (U <-> U)
                    # Канонизация и дедупликация (a, b) и (b, a)
                    canonical_edges = set(tuple(sorted((u, v))) for u, v in edges)
                    final_edges = list(canonical_edges)
                    num_logical_edge_combinations += 1
                    
                else: 
                    # Case 3: Ненаправленная межгрупповая связь (U <-> V)
                    
                    # 1. Собираем все ребра U->V и V->U
                    canonical_edges = set()
                    
                    # U -> V ребра (прямое направление)
                    for u_local, v_local in edges:
                        canonical_edge = tuple(sorted((u_local, v_local)))
                        canonical_edges.add(canonical_edge)
                    
                    # V -> U ребра (обратное направление)
                    reverse_u_group, reverse_v_group = v_group, u_group
                    reverse_edges = edge_lists.get(reverse_u_group, {}).get(reverse_v_group, [])
                    for v_local, u_local in reverse_edges:
                        canonical_edge = tuple(sorted((u_local, v_local))) 
                        canonical_edges.add(canonical_edge)
                    
                    final_edges = list(canonical_edges)
                    
                    # 2. Сохраняем результат для обеих сторон, используя ССЫЛКУ на один массив
                    
                    # Прямое направление (key)
                    result_array = np.array(final_edges, dtype=np.int32) if final_edges else np.empty((0, 2), dtype=np.int32)
                    self.results[key] = result_array
                    num_logical_edge_combinations += 1
                    
                    # Обратное направление (reverse_key) - ССЫЛКА НА ТОТ ЖЕ МАССИВ!
                    reverse_key = f"edges_{reverse_u_group}_to_{reverse_v_group}"
                    self.results[reverse_key] = self.results[key]
                    
                    # 3. Добавляем обратное направление в список пропущенных для следующей итерации
                    processed_undirected_pairs.add((reverse_u_group, reverse_v_group))
                    
                    # Поскольку результат уже сохранен, переходим к следующей итерации
                    continue 

                # Сохранение результата для Case 1 и Case 2
                if not final_edges:
                    self.results[key] = np.empty((0, 2), dtype=np.int32)
                else:
                    self.results[key] = np.array(final_edges, dtype=np.int32)
            
        print(f"Сгенерированы массивы ребер для {num_logical_edge_combinations} логических комбинаций.")


    def get_results(self) -> Dict[str, Any]:
        """
        Возвращает словарь, содержащий все сгенерированные структуры данных.
        
        Содержит:
        - 'reindexed_graph': nx.DiGraph с глобальными индексами (0..N-1). (Для внутреннего использования)
        - 'mapping': Словарь с маппингом old_id <-> new_global_id. (Для внутреннего использования)
        - 'num_nodes': Словарь с количеством узлов по группам.
        - 'edges_{group1}_to_{group2}': Массивы np.array(N, 2) с локальными индексами ребер.
        """
        return self.results

    def save_to_npz(self, filepath: str):
        """
        Сохраняет массивы ребер и количество узлов в сжатый файл NPZ.

        Args:
            filepath: Путь к файлу для сохранения (например, 'graph_data.npz').
        """
        data_to_save = {}
        
        # 1. Сохранение массивов ребер
        for key, value in self.results.items():
            if key.startswith('edges_'):
                data_to_save[key] = value

        # 2. Сохранение количества узлов как отдельных скалярных массивов
        num_nodes = self.results.get('num_nodes', {})
        for group, count in num_nodes.items():
            data_to_save[f"num_{group}"] = np.array(count, dtype=np.int32)
            
        np.savez_compressed(filepath, **data_to_save)
        print(f"Данные графа успешно сохранены в {filepath} (ключи: {list(data_to_save.keys())})")

    @staticmethod
    def load_from_npz(filepath: str) -> Dict[str, Any]:
        """
        Загружает массивы ребер и количество узлов из NPZ файла и возвращает словарь.

        Args:
            filepath: Путь к NPZ файлу.

        Returns:
            Словарь, содержащий 'num_nodes' и массивы ребер 'edges_...'.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл не найден: {filepath}")

        loaded_data = np.load(filepath)
        results = {}
        num_nodes = {}

        for key in loaded_data.files:
            if key.startswith('edges_'):
                results[key] = loaded_data[key]
            elif key.startswith('num_'):
                # Ключи 'num_H', 'num_S'
                group_name = key[4:]
                # Извлекаем скалярное значение из 0D массива (формат сохранения np.savez)
                num_nodes[group_name] = int(loaded_data[key].item()) 

        results['num_nodes'] = num_nodes
        print(f"Данные успешно загружены из {filepath}. Количество узлов: {num_nodes}")
        return results