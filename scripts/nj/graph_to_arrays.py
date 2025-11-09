import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Union, Set
import os
import hashlib
import json
import itertools

# --- Типы данных ---
# DirectednessMap: Словарь, определяющий, является ли связь между двумя группами направленной (True) или нет (False).
DirectednessMap = Dict[str, Dict[str, bool]] 
# InitialValueMap: Словарь для конфигурации начального состояния (скаляр или размерность признака).
InitialValueMap = Dict[str, Union[float, int]]

# ==============================================================================
# 1. JaxGraphConverter
# ==============================================================================

class JaxGraphConverter:
    """
    Преобразует граф NetworkX с произвольными ID и типами узлов 
    в наборы массивов ребер и карт маппинга, готовых для использования в JAX GNN.
    
    Ключевая цель: переиндексация узлов в последовательные локальные индексы (0..N-1) 
    внутри каждой группы узлов (напр., 'H', 'S').
    """

    def __init__(self,
                 graph: Union[nx.DiGraph, nx.MultiDiGraph],
                 node_type_groups: Dict[str, List[str]],
                 edge_directedness: DirectednessMap):
        
        self.graph = graph
        self.node_type_groups = node_type_groups
        self.edge_directedness = edge_directedness
        
        # Результаты конвертации
        self.num_nodes: Dict[str, int] = {}             # {'H': 5, 'S': 3}
        self.edge_arrays: Dict[str, np.ndarray] = {}    # {'edges_H_to_H': ..., 'edges_H_to_S': ...}
        
        # Карты маппинга:
        # 1. Глобальный маппинг: old_id (Any) -> new_global_id (int 0..TotalN-1)
        self.global_mapping: Dict[Any, int] = {}
        # 2. Локальный маппинг: new_global_id (int) -> new_local_id (int 0..N_group-1)
        self.local_maps: Dict[str, Dict[int, int]] = {} 

        self._process_graph()

    @staticmethod
    def load_from_npz(filepath: str) -> Dict[str, Any]:
        """
        Загружает результаты конвертации из NPZ-файла.
        ВАЖНО: Должен быть реализован и в Converter, и в Context для кэширования.
        """
        data = np.load(filepath, allow_pickle=True)
        results: Dict[str, Any] = {}
        num_nodes: Dict[str, int] = {}
        
        for key in data.files:
            if key.startswith('num_'):
                num_nodes[key.split('_')[1]] = data[key].item() # Извлечение скалярного значения
            elif key.startswith('edges_'):
                results[key] = data[key]
            # Загружаем карты маппинга, которые были сохранены как объекты
            elif key == 'local_maps':
                results['local_maps'] = data[key].item() # .item() для извлечения словаря
            elif key == 'global_mapping':
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
        # Сохраняем local_maps
        data_to_save['local_maps'] = np.array(self.local_maps, dtype=object)
        
        # Сохраняем global_mapping под другим именем для удобства загрузки
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
            **self.edge_arrays  # Распаковка массивов ребер
        }

    def _process_graph(self):
        """
        Основной метод обработки: 
        1. Создание глобального и локального маппингов.
        2. Генерация массивов ребер.
        """
        # --- 1. Группировка узлов и глобальный/локальный маппинг ---
        self._create_mappings()
        print(f"Узлы успешно сгруппированы: {self.num_nodes}")

        # --- 2. Генерация массивов ребер ---
        self._create_edge_arrays()
        print(f"Сгенерированы массивы ребер для {len(self.edge_arrays)} логических комбинаций.")


    def _get_group_name(self, node_id: Any) -> Union[str, None]:
        """Определяет группу узла по его атрибуту 'type'."""
        node_type = self.graph.nodes[node_id].get('type')
        if not node_type:
            return None # Пропускаем узлы без типа
        
        for group_name, type_list in self.node_type_groups.items():
            if node_type in type_list:
                return group_name
        return None

    def _create_mappings(self):
        """Создает карты маппинга старых ID в новые глобальные и локальные индексы."""
        
        # 1. Сортировка узлов и создание глобального маппинга (old_id -> new_global_id)
        # Сначала сортируем все узлы по их оригинальному ID (строковое представление для детерминизма)
        sorted_nodes = sorted(self.graph.nodes())
        
        # Создаем глобальный маппинг (0 до TotalN-1)
        global_idx = 0
        for node_id in sorted_nodes:
            self.global_mapping[node_id] = global_idx
            global_idx += 1
            
        # 2. Создание локального маппинга (new_global_id -> new_local_id)
        # Инициализируем локальные маппинги и счетчики узлов
        self.local_maps = {group: {} for group in self.node_type_groups.keys()}
        self.num_nodes = {group: 0 for group in self.node_type_groups.keys()}

        # Обрабатываем узлы снова, чтобы назначить локальные ID
        # Важно: Сортировка здесь снова нужна для детерминизма локальных ID
        for node_id in sorted_nodes:
            group_name = self._get_group_name(node_id)
            if group_name and group_name in self.local_maps:
                global_id = self.global_mapping[node_id]
                local_idx = self.num_nodes[group_name]
                
                self.local_maps[group_name][global_id] = local_idx
                self.num_nodes[group_name] += 1
                
        # Удаляем группы, в которых нет узлов
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
            
            # Находим локальные маппинги для исходной и целевой групп
            src_local_map = self.local_maps.get(src_group, {})
            dst_local_map = self.local_maps.get(dst_group, {})

            if not src_local_map or not dst_local_map:
                continue # Пропускаем, если группа пуста

            for u_old, v_old in self.graph.edges():
                # 1. Определяем группы исходного и целевого узлов (по оригинальным ID)
                u_group = self._get_group_name(u_old)
                v_group = self._get_group_name(v_old)
                
                # 2. Проверяем, соответствует ли ребро текущей паре групп
                if u_group == src_group and v_group == dst_group:
                    
                    # 3. Преобразуем старые ID в глобальные ID
                    u_global_id = self.global_mapping[u_old]
                    v_global_id = self.global_mapping[v_old]
                    
                    # 4. Преобразуем глобальные ID в локальные индексы
                    u_local_idx = src_local_map.get(u_global_id)
                    v_local_idx = dst_local_map.get(v_global_id)
                    
                    if u_local_idx is not None and v_local_idx is not None:
                        edge_list.append((u_local_idx, v_local_idx))

            # Обработка ненаправленных ребер (когда True в edge_directedness означает направленность)
            if not is_directed:
                # Если связь A->B ненаправленная, добавляем B->A. 
                # Важно: src_group и dst_group должны быть одинаковыми в этом случае (A->A, B->B).
                # В противном случае, это симметричное ребро (A->B и B->A) должно быть 
                # обработано как A->B и B->A в соответствующих итерациях цикла product.
                
                # Упрощение: В GNN для ненаправленной связи X-Y мы обычно добавляем (X->Y) и (Y->X) 
                # к массиву ребер, соответствующему типу X->Y (если X=Y).
                # Здесь мы предполагаем, что ненаправленность используется только для однотипных связей (H->H)
                if src_group == dst_group:
                    undirected_edges = set(edge_list)
                    for u, v in edge_list:
                        undirected_edges.add((v, u))
                    edge_list = list(undirected_edges)
                
            
            if edge_list:
                # Массивы ребер должны быть (2, Num_Edges) для JAX
                array = np.array(edge_list, dtype=np.int32).T 
                key = f'edges_{src_group}_to_{dst_group}'
                self.edge_arrays[key] = array
                

# ==============================================================================
# 2. SimulationContextJax (Исправленный)
# ==============================================================================

class SimulationContextJax:
    """
    Класс-контейнер для подготовки всех данных, необходимых для GNN симуляции в JAX:
    - Массивы ребер (с учетом направленности и дедупликации).
    - Количество узлов по группам.
    - Начальные массивы состояний/признаков узлов.
    - Поддержка кэширования структур данных графа для ускорения повторных запусков.
    """

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
        
        # 1. Запуск логики кэширования
        cache_hit = False
        if self.cache_dir:
            self.graph_hash = self._calculate_graph_hash(graph, node_type_groups, edge_directedness)
            cache_hit = self._try_load_from_cache()

        if cache_hit:
            print("1. Данные графа успешно загружены из кэша.")
        else:
            print("1. Кэш пропущен/не найден. Запуск JaxGraphConverter для обработки графа...")
            # При кэш-миссе/отсутствии кэша - запускаем конвертацию
            converter = JaxGraphConverter(graph, node_type_groups, edge_directedness)
            self.graph_results = converter.get_results()
            
            # Сохраняем результат в кэш
            if self.cache_dir:
                self._save_to_cache(converter)
        
        # FIX: Убеждаемся, что все нужные ключи извлечены
        # 'local_maps' теперь гарантированно присутствует в self.graph_results
        self.num_nodes = self.graph_results['num_nodes']
        print(self.graph_results)
        self.local_maps = self.graph_results['local_maps']
        
        self.initial_states: Dict[str, np.ndarray] = {}

        print("2. Инициализация начальных массивов состояний узлов...")
        self._initialize_node_states()

        print("Контекст симуляции JAX готов.")

    @staticmethod
    def _calculate_graph_hash(graph: Union[nx.DiGraph, nx.MultiDiGraph], node_type_groups: Dict, edge_directedness: Dict) -> str:
        """
        Генерирует SHA256 хэш на основе канонической структуры графа и конфигурации конвертера.
        """
        # 1. Хэширование структуры графа (узлы + ребра + атрибуты 'type')
        
        node_data = sorted([(str(n), graph.nodes[n].get('type')) for n in graph.nodes()])
        node_str = json.dumps(node_data, sort_keys=True)

        edge_data = sorted([(str(u), str(v)) for u, v in graph.edges()])
        edge_str = json.dumps(edge_data, sort_keys=True)
        
        # 2. Хэширование конфигурации конвертера
        config_data = {
            "groups": node_type_groups,
            "directedness": edge_directedness
        }
        config_str = json.dumps(config_data, sort_keys=True)

        full_string = (node_str + edge_str + config_str).encode('utf-8')
        return hashlib.sha256(full_string).hexdigest()

    def _get_cache_paths(self) -> Tuple[str, str]:
        """Возвращает полные пути к файлам NPZ и HASH в директории кэша."""
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
            print(f"   Кэш не найден по пути: {os.path.dirname(npz_path)}")
            return False
            
        try:
            with open(hash_path, 'r') as f:
                saved_hash = f.read().strip()
                
            if saved_hash == self.graph_hash:
                print(f"   Хэш совпадает: {self.graph_hash[:10]}...")
                # Используем статический метод JaxGraphConverter для загрузки NPZ
                self.graph_results = JaxGraphConverter.load_from_npz(npz_path)
                return True
            else:
                print(f"   Хэш не совпадает (старый: {saved_hash[:10]}..., новый: {self.graph_hash[:10]}...). Перерасчет.")
                return False
                
        except Exception as e:
            print(f"Ошибка при чтении кэша: {e}. Перерасчет.")
            return False

    def _save_to_cache(self, converter: JaxGraphConverter):
        """Сохраняет сгенерированные массивы и хэш в кэш директорию."""
        try:
            npz_path, hash_path = self._get_cache_paths()
            
            # 1. Сохранение данных NPZ
            converter.save_to_npz(npz_path)
            
            # 2. Сохранение хэша
            with open(hash_path, 'w') as f:
                f.write(self.graph_hash)
            
            print(f"   Данные графа и хэш успешно сохранены в кэш: {os.path.dirname(npz_path)}")
            
        except Exception as e:
            print(f"Внимание: Не удалось сохранить данные в кэш: {e}")

    def _initialize_node_states(self):
        """
        Создает массивы NumPy для начальных состояний узлов на основе
        конфигурации self.initial_node_values.
        """
        for group_name, num_n in self.num_nodes.items():
            
            if group_name not in self.initial_node_values:
                print(f"Внимание: Для группы '{group_name}' не задано начальное значение. Пропуск.")
                continue

            value = self.initial_node_values[group_name]
            
            if isinstance(value, (int, float)) and value != 0: 
                if value <= 1:
                    # Case 1: Скалярное значение (float/int, включая 1). 
                    initial_value = float(value)
                    shape = (num_n,)
                    self.initial_states[group_name] = np.full(shape, initial_value, dtype=np.float32)
                    print(f"   > '{group_name}': форма {shape}, заполнена {initial_value}")
                elif isinstance(value, int) and value > 1:
                    # Case 2: int > 1. Интерпретируем как размерность вектора признаков (D).
                    dimension = value
                    shape = (num_n, dimension)
                    self.initial_states[group_name] = np.zeros(shape, dtype=np.float32)
                    print(f"   > '{group_name}': форма {shape}, инициализирована нулями (размерность признаков {dimension})")
                else:
                    print(f"   > '{group_name}': Пропущено. Скалярное значение > 1 должно быть целым числом для размерности: {value}.")
            else:
                 print(f"   > '{group_name}': Пропущено. Некорректное начальное значение (должно быть != 0): {value}.")


    def get_context(self) -> Dict[str, Any]:
        """
        Возвращает полный словарь контекста, готовый для использования в GNN модели.
        """
        context = {
            'num_nodes': self.num_nodes,
            'initial_states': self.initial_states,
            # Добавляем все массивы ребер, сгенерированные конвертером
            **{k: v for k, v in self.graph_results.items() if k.startswith('edges_')}
        }
        return context

    def get_node_id_mapping(self) -> Dict[str, Dict[Any, int]]:
        """
        Возвращает маппинг старых (оригинальных) ID узлов в новые (локальные) индексы
        для каждой группы узлов.

        Returns:
            Словарь, где ключ - имя группы ('H', 'S', и т.д.), 
            а значение - словарь: {оригинальный_id: локальный_индекс}.
        """
        # Получаем глобальные маппинги (old -> new_global)
        global_map = self.graph_results.get('mapping', {}).get('old_to_new_global', {})
        
        # Получаем локальные маппинги (new_global -> new_local) - теперь это self.local_maps
        local_maps_global_to_local = self.local_maps
        
        final_mapping = {}
        
        # Инвертируем локальные карты для удобства: new_local_id -> new_global_id
        # Но для нашей цели (old_id -> new_local_id) лучше перебирать global_map
        
        # 1. Создаем обратный маппинг: new_global_id -> group_name
        global_to_group: Dict[int, str] = {}
        for group_name, map_dict in local_maps_global_to_local.items():
            for global_id, local_id in map_dict.items():
                global_to_group[global_id] = group_name

        # 2. Создаем конечный маппинг
        for group_name in self.num_nodes.keys():
            final_mapping[group_name] = {}

        for original_id, global_id in global_map.items():
            group_name = global_to_group.get(global_id)
            
            if group_name and group_name in final_mapping:
                # Получаем локальный индекс из карты local_maps
                local_index = self.local_maps[group_name].get(global_id)
                if local_index is not None:
                    final_mapping[group_name][original_id] = local_index
            
        return final_mapping


# --- Демонстрация использования ---
if __name__ == '__main__':
    
    # ВРЕМЕННЫЙ КЭШ ДИРЕКТОРИЯ
    CACHE_DIR = './jax_context_cache' 
    import shutil

    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print(f"Удалена старая тестовая директория кэша '{CACHE_DIR}'.")
    
    # 1. Создание примера графа NetworkX с нестандартными ID
    G = nx.DiGraph() 
    original_ids = [1001, 1005, 2010, 2015, 1020, 1025, 2030, 2035]
    node_configs = {
        1001: 'H_root', 1005: 'H_root', 
        2010: 'S_conn', 2015: 'H_root', 
        1020: 'S_conn', 1025: 'H_root',
        2030: 'H_root', 2035: 'S_conn',
    }
    for oid, ntype in node_configs.items():
        G.add_node(oid, type=ntype)
    
    # Добавляем ребра
    G.add_edges_from([
        (1001, 1005), # H->H
        (1005, 1001), # H->H (Undirected in config, so both should be kept)
        (2010, 1001), # S->H
        (1005, 1020), # H->S
        (2035, 2010), # S->S
        (1025, 2030)  # H->H
    ])
    
    # 2. Конфигурация
    type_groups = {
        'H': ['H_root'],        
        'S': ['S_conn']         
    }
    # H-H - ненаправленная (False), H-S и S-H - направленные (True), S-S - направленная (True)
    directedness = {
        'H': {'H': False, 'S': True},
        'S': {'H': True, 'S': True}
    }
    initial_values_config = {
        'H': 0.5, # Скалярное начальное значение
        'S': 3    # Вектор признаков размерности 3
    }
    
    # --- Тест 1: Кэш-мисс ---
    print("\n================== ТЕСТ 1: КЭШ-МИСС (ПЕРВЫЙ ЗАПУСК) ==================")
    context_manager = SimulationContextJax(
        graph=G,
        node_type_groups=type_groups,
        edge_directedness=directedness,
        initial_node_values=initial_values_config,
        cache_dir=CACHE_DIR 
    )
    
    # 3. Получение карты маппинга
    mapping_dict = context_manager.get_node_id_mapping()
    
    print("\n--- Карта маппинга (оригинальный ID -> локальный индекс) ---")
    print("Группа H (H_root):", mapping_dict.get('H'))
    print("Группа S (S_conn):", mapping_dict.get('S'))

    # 4. Пример DataFrame с дополнительными данными
    df_data = pd.DataFrame({
        'original_id': [1001, 2010, 1025, 2035, 1005],
        'node_type': ['H_root', 'S_conn', 'H_root', 'S_conn', 'H_root'],
        'feature_A': [10.5, 20.1, 15.0, 5.5, 99.9]
    })
    
    print("\n--- Исходный DataFrame ---")
    print(df_data)

    # 5. Сопоставление DataFrame с локальными индексами
    def map_to_local_index(row):
        group_map = {'H_root': 'H', 'S_conn': 'S'}
        group = group_map.get(row['node_type'])
        if not group:
            return None
            
        original_id_str = str(row['original_id']) 
        local_index = mapping_dict.get(group, {}).get(original_id_str)
        return local_index

    df_data['local_index'] = df_data.apply(map_to_local_index, axis=1)
    
    print("\n--- DataFrame с добавленным 'local_index' ---")
    print(df_data[['original_id', 'node_type', 'local_index', 'feature_A']])

    # 6. Проверка загрузки из кэша
    print("\n================== ТЕСТ 2: КЭШ-ХИТ (ВТОРОЙ ЗАПУСК) ==================")
    # Удаляем объект и создаем его снова
    del context_manager 
    
    context_manager_2 = SimulationContextJax(
        graph=G,
        node_type_groups=type_groups,
        edge_directedness=directedness,
        initial_node_values=initial_values_config,
        cache_dir=CACHE_DIR 
    )

    # Проверка, что local_maps доступен и здесь
    mapping_dict_2 = context_manager_2.get_node_id_mapping()
    print("\n--- Карта маппинга после загрузки из Кэша (H-Group) ---")
    print(mapping_dict_2.get('H'))

    # --- Очистка кэша ---
    try:
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
    except Exception as e:
        pass