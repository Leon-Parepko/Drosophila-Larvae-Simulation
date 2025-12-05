import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Set
import os
import itertools

# --- Типы данных ---
DirectednessMap = Dict[str, Dict[str, bool]] 
GraphResults = Dict[str, Any]
# GraphResults теперь включает: num_nodes, mapping, edges_..., features_...,
# а также скрытые ключи: __RAW_GLOBAL_MAPPING__ и __RAW_LOCAL_MAPS__

# ==============================================================================
# 1. Вспомогательные функции (Логика конвертации)
# ==============================================================================

def _get_group_name(graph: Union[nx.DiGraph, nx.MultiDiGraph], node_id: Any, node_type_groups: Dict[str, List[str]]) -> List[str]:
    """
    Определяет группы узла по его атрибуту 'type'.
    Возвращает список имен групп, к которым принадлежит узел.
    """
    node_type = graph.nodes[node_id].get('type')
    if not node_type:
        return []
    
    found_groups: List[str] = []
    for group_name, type_list in node_type_groups.items():
        if node_type in type_list:
            found_groups.append(group_name)
    return found_groups

def _create_mappings(graph: Union[nx.DiGraph, nx.MultiDiGraph], node_type_groups: Dict[str, List[str]]) -> Tuple[Dict[Any, int], Dict[str, Dict[int, int]], Dict[str, int]]:
    """
    Создает карты маппинга старых ID в новые глобальные и локальные индексы.
    
    Возвращает: global_mapping (old_id -> global_id), 
                local_maps (global_id -> local_id), 
                num_nodes (group_name -> count)
    """
    sorted_nodes = sorted(graph.nodes())
    
    # 1. Глобальный маппинг
    global_mapping: Dict[Any, int] = {}
    global_idx = 0
    for node_id in sorted_nodes:
        global_mapping[node_id] = global_idx
        global_idx += 1
        
    # 2. Локальные маппинги и счетчики
    local_maps: Dict[str, Dict[int, int]] = {group: {} for group in node_type_groups.keys()}
    num_nodes: Dict[str, int] = {group: 0 for group in node_type_groups.keys()}

    for node_id in sorted_nodes:
        group_names = _get_group_name(graph, node_id, node_type_groups)
        
        for group_name in group_names:
            if group_name in local_maps:
                global_id = global_mapping[node_id]
                
                # Присваиваем локальный ID
                if global_id not in local_maps[group_name]:
                    local_idx = num_nodes[group_name]
                    local_maps[group_name][global_id] = local_idx
                    num_nodes[group_name] += 1
            
    # Удаляем пустые группы
    num_nodes = {k: v for k, v in num_nodes.items() if v > 0}
    local_maps = {k: v for k, v in local_maps.items() if v}
    
    return global_mapping, local_maps, num_nodes

def _create_edge_arrays(graph: Union[nx.DiGraph, nx.MultiDiGraph], 
                        global_mapping: Dict[Any, int], 
                        local_maps: Dict[str, Dict[int, int]], 
                        num_nodes: Dict[str, int], 
                        edge_directedness: DirectednessMap, 
                        node_type_groups: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    """
    Создает массивы ребер (src, dst) с использованием локальных индексов.
    """
    edge_arrays: Dict[str, np.ndarray] = {}
    groups = list(num_nodes.keys())
    
    for src_group, dst_group in itertools.product(groups, groups):
        edge_list: List[Tuple[int, int]] = []
        is_directed = edge_directedness.get(src_group, {}).get(dst_group, True)
        
        src_local_map = local_maps.get(src_group, {})
        dst_local_map = local_maps.get(dst_group, {})

        if not src_local_map or not dst_local_map:
            continue 

        for u_old, v_old in graph.edges():
            u_groups = _get_group_name(graph, u_old, node_type_groups)
            v_groups = _get_group_name(graph, v_old, node_type_groups)
            
            # Ищем совпадение групп
            found_match = False
            for u_group in u_groups:
                for v_group in v_groups:
                    if u_group == src_group and v_group == dst_group:
                        u_global_id = global_mapping[u_old]
                        v_global_id = global_mapping[v_old]
                        
                        u_local_idx = src_local_map.get(u_global_id)
                        v_local_idx = dst_local_map.get(v_global_id)
                        
                        if u_local_idx is not None and v_local_idx is not None:
                            edge_list.append((u_local_idx, v_local_idx))
                            found_match = True
                            break 
                if found_match:
                    break 

        if not is_directed:
            # Логика для создания симметричных ребер
            if src_group == dst_group:
                undirected_edges = set(edge_list)
                for u, v in edge_list:
                    undirected_edges.add((v, u))
                edge_list = list(undirected_edges)
            
        
        if edge_list:
            unique_edges = sorted(list(set(edge_list)))
            # Transpose to get [src_indices, dst_indices] shape
            array = np.array(unique_edges, dtype=np.int32).T 
            key = f'edges_{src_group}_to_{dst_group}'
            edge_arrays[key] = array
            
    return edge_arrays

def _calculate_user_mapping(
    num_nodes: Dict[str, int], 
    global_map_old_to_new: Dict[Any, int],
    local_maps_global_to_local: Dict[str, Dict[int, int]]
) -> Dict[str, Dict[Any, int]]:
    """
    Рассчитывает конечный пользовательский маппинг: {group_name: {old_id: local_id}}.
    """
    final_mapping: Dict[str, Dict[Any, int]] = {k: {} for k in num_nodes.keys()}
    
    for original_id, global_id in global_map_old_to_new.items():
        # Перебираем все группы, чтобы найти local_id для этого global_id
        for group_name in num_nodes.keys():
            local_map = local_maps_global_to_local.get(group_name, {})
            local_index = local_map.get(global_id)
            
            if local_index is not None:
                # Узел принадлежит этой группе, добавляем его в маппинг
                final_mapping[group_name][original_id] = local_index
    
    return final_mapping


# ==============================================================================
# 2. Основные публичные функции (Обработка и Сохранение)
# ==============================================================================

def process_graph_to_core_arrays(
    graph: Union[nx.DiGraph, nx.MultiDiGraph], 
    node_type_groups: Dict[str, List[str]], 
    edge_directedness: DirectednessMap
) -> GraphResults:
    """
    Принимает граф и конфигурацию, обрабатывает его и возвращает полный словарь 
    GraphResults, который готов к использованию в JAX GNN и к сохранению.
    
    Включает скрытые raw-маппинги, необходимые для I/O.

    Returns:
        GraphResults: Словарь, содержащий num_nodes, mapping, edge_arrays и raw-маппинги.
    """
    print("Начало обработки графа (маппинги и ребра)...")
    
    # 1. Создание raw-маппингов и счетчиков
    global_mapping, local_maps, num_nodes = _create_mappings(graph, node_type_groups)
    
    # 2. Создание массивов ребер
    edge_arrays = _create_edge_arrays(graph, global_mapping, local_maps, num_nodes, edge_directedness, node_type_groups)
    
    # 3. Расчет пользовательского маппинга
    final_mapping = _calculate_user_mapping(num_nodes, global_mapping, local_maps)
    
    print(f"Обработка завершена. Найдено {len(num_nodes)} групп и {len(edge_arrays)} массивов ребер.")
    
    # 4. Сборка полного контекста
    context: GraphResults = {
        # Основной контекст для GNN
        'num_nodes': num_nodes,
        'mapping': final_mapping,
        **edge_arrays, # Включает edges_...

        # Скрытые raw-маппинги (необходимы для save/load, не для GNN-модели)
        '__RAW_GLOBAL_MAPPING__': global_mapping,
        '__RAW_LOCAL_MAPS__': local_maps,
    }
    
    return context


def extract_node_features(
    graph: Union[nx.DiGraph, nx.MultiDiGraph], 
    context: GraphResults, 
    feature_config: Dict[str, List[str]]
) -> Dict[str, np.ndarray]:
    """
    Извлекает указанные свойства узлов из графа и сохраняет их в массивы NumPy, 
    используя локальную индексацию, извлеченную из контекста.
    
    ВАЖНО: Принимает полный контекст, но извлекает из него только необходимые raw-маппинги.
        
    Returns:
        Dict[str, np.ndarray]: Словарь с массивами признаков вида {'features_<group_name>': array}.
    """
    feature_arrays: Dict[str, np.ndarray] = {}
    
    if not feature_config:
        return feature_arrays
    
    # Извлечение raw-маппингов и счетчиков из контекста
    global_mapping = context['__RAW_GLOBAL_MAPPING__']
    local_maps = context['__RAW_LOCAL_MAPS__']
    num_nodes = context['num_nodes']

    # Проходим по каждой группе узлов, для которой определены свойства
    for group_name, properties_list in feature_config.items():
        if group_name not in num_nodes or not properties_list:
            continue 

        num = num_nodes[group_name]
        local_map = local_maps.get(group_name, {})
        
        num_features = len(properties_list)
        # Используем float64 для универсальности
        feature_array = np.zeros((num, num_features), dtype=np.float64) 
        
        # Проходим по всем узлам графа
        for node_id, global_id in global_mapping.items():
            
            local_idx = local_map.get(global_id)
            if local_idx is None:
                continue

            node_data = graph.nodes[node_id]
            
            # Извлекаем и заполняем каждое свойство
            for prop_idx, prop_key in enumerate(properties_list):
                value = node_data.get(prop_key, 0.0)
                try:
                    feature_array[local_idx, prop_idx] = float(value)
                except (ValueError, TypeError):
                    feature_array[local_idx, prop_idx] = 0.0
                    
        key = f'features_{group_name}'
        feature_arrays[key] = feature_array

    return feature_arrays


def save_jax_arrays(
    context: GraphResults,
    save_path: str,
    additional_data: Dict[str, np.ndarray] = None # Дополнительные произвольные массивы
):
    """
    Сохраняет полный контекст графа в NPZ-файл.
    
    Args:
        context: Полный словарь GraphResults (включает num_nodes, mapping, edges_... и raw-маппинги).
        save_path: Путь для сохранения.
        additional_data: Словарь произвольных массивов NumPy для включения в файл.
    """
    print(f"Начало сохранения данных в {save_path}...")
    
    # Проверка наличия обязательных raw-маппингов
    global_mapping = context.get('__RAW_GLOBAL_MAPPING__')
    local_maps = context.get('__RAW_LOCAL_MAPS__')
    num_nodes = context.get('num_nodes')

    if global_mapping is None or local_maps is None:
        raise ValueError("В контексте отсутствуют обязательные raw-маппинги (__RAW_GLOBAL_MAPPING__ или __RAW_LOCAL_MAPS__). Убедитесь, что контекст создан с помощью process_graph_to_core_arrays.")
        
    data_to_save: Dict[str, Any] = {}
    
    # 1. Количество узлов
    for group, count in num_nodes.items():
        data_to_save[f'num_{group}'] = np.array(count)
        
    # 2. Все массивы (ребра, фичи, дополнительно) из контекста
    for key, value in context.items():
        if isinstance(value, np.ndarray):
            data_to_save[key] = value

    # 3. Дополнительные массивы (для случаев, когда они были созданы вне контекста)
    if additional_data:
        data_to_save.update(additional_data)
        
    # 4. Карты маппинга (для восстановления контекста при загрузке)
    data_to_save['local_maps'] = np.array(local_maps, dtype=object)
    data_to_save['global_mapping'] = np.array(global_mapping, dtype=object)
    
    # 5. Сохранение
    num_saved_arrays = len(data_to_save) - 3 - len(num_nodes) # -3 за маппинги
    np.savez_compressed(save_path, **data_to_save)
    print(f"Сохранение завершено. Всего {num_saved_arrays} массивов (ребер + дополнительных) и метаданные сохранены.")


def load_jax_context(load_path: str) -> GraphResults:
    """
    Загружает обработанную структуру графа из NPZ-файла и формирует полный контекст.

    Returns:
        GraphResults: Словарь, содержащий:
                      - 'num_nodes': Dict[str, int]
                      - 'mapping': Dict[str, Dict[Any, int]] (group_name -> {old_id -> local_id})
                      - 'edges_...': np.ndarray (массивы ребер)
                      - <любой другой массив>
                      - __RAW_GLOBAL_MAPPING__ и __RAW_LOCAL_MAPS__
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Файл данных не найден: {load_path}")
        
    print(f"Загрузка контекста из: {load_path}")
    
    data = np.load(load_path, allow_pickle=True)
    
    num_nodes: Dict[str, int] = {}
    array_results: Dict[str, np.ndarray] = {} 
    
    global_map_old_to_new: Dict[Any, int] = {}
    local_maps_global_to_local: Dict[str, Dict[int, int]] = {}

    for key in data.files:
        if key.startswith('num_'):
            num_nodes[key.split('_')[1]] = data[key].item() 
        elif key == 'local_maps':
            local_maps_global_to_local = data[key].item() 
        elif key == 'global_mapping':
            global_map_old_to_new = data[key].item()
        else:
            # Все остальное - массивы данных (edges_..., features_...)
            array_results[key] = data[key]
            
    num_edges = len([k for k in array_results if k.startswith('edges_')])
    num_features = len([k for k in array_results if k.startswith('features_')])

    print(f"   Загружено: {len(num_nodes)} групп, {num_edges} массивов ребер, {num_features} массивов признаков.")

    # 1. Расчет пользовательского маппинга
    final_mapping = _calculate_user_mapping(
        num_nodes, 
        global_map_old_to_new, 
        local_maps_global_to_local
    )

    # 2. Сборка конечного контекста
    context = {
        'num_nodes': num_nodes,
        'mapping': final_mapping, 
        **array_results,
        
        # Скрытые raw-маппинги для консистентности и повторного сохранения
        '__RAW_GLOBAL_MAPPING__': global_map_old_to_new,
        '__RAW_LOCAL_MAPS__': local_maps_global_to_local,
    }
    
    print("Контекст симуляции успешно собран из загруженных данных.")
    return context


# ==============================================================================
# Демонстрация использования (опционально)
# ==============================================================================

if __name__ == '__main__':
    
    # --- Конфигурация для теста ---
    SAVE_FILE = 'test_graph_arrays.npz'
    
    # 1. Создание примера графа NetworkX
    G = nx.DiGraph() 
    
    type_groups = {'H': ['H_root', 'HS_hybrid'], 'S': ['S_conn', 'HS_hybrid']}
    
    node_configs = {
        1001: ('H_root', {'bias': 0.1, 'weight': 5.0}), 
        1005: ('H_root', {'bias': 0.5, 'weight': 10.0}), 
        2010: ('S_conn', {'bias': 0.9, 'initial_value': 2.0}), 
        9000: ('HS_hybrid', {'bias': 0.2, 'weight': 8.0, 'initial_value': 1.5}) 
    }
    for oid, (ntype, attrs) in node_configs.items():
        G.add_node(oid, type=ntype, **attrs)
    
    G.add_edges_from([
        (1001, 1005), 
        (9000, 1001), 
        (2010, 9000), 
        (1005, 2010),
    ])
    
    directedness = {'H': {'H': False, 'S': True}, 'S': {'H': True, 'S': True}}
    
    # Конфигурация признаков для извлечения
    feature_config_H = {'H': ['bias', 'weight']}
    feature_config_S = {'S': ['initial_value', 'bias']}

    print("================== ТЕСТ 1: ОБРАБОТКА (возвращает единый контекст) ==================")
    
    # 1. Получаем единый контекст
    context = process_graph_to_core_arrays(
        graph=G,
        node_type_groups=type_groups,
        edge_directedness=directedness
    )
    
    # 2. Извлечение признаков (теперь просто обновляем контекст)
    features_H = extract_node_features(G, context, feature_config_H)
    features_S = extract_node_features(G, context, feature_config_S)
    
    # 3. Произвольные данные (абстрактные маски)
    # Предположим, что H имеет 3 узла, а S - 2 узла (по данным num_nodes)
    custom_mask = {'mask_H': np.array([1, 0, 1], dtype=np.int32), 
                   'mask_S': np.array([0, 1], dtype=np.int32)} 

    # 4. Включаем все дополнительные данные в контекст
    context.update(features_H)
    context.update(features_S)
    context.update(custom_mask)
    
    # 5. Сохранение: просто передаем единый контекст
    print("\n================== ТЕСТ 2: СОХРАНЕНИЕ (принимает единый контекст) ==================")
    save_jax_arrays(
        context=context,
        save_path=SAVE_FILE,
        # additional_data здесь не нужен, так как все добавлено в context
    )
    
    print("\n================== ТЕСТ 3: ЗАГРУЗКА И ПРОВЕРКА КОНТЕКСТА ==================")
    
    # 6. Загрузка
    loaded_context = load_jax_context(SAVE_FILE)
    
    # Проверка структуры загруженного контекста
    print("\nСтруктура загруженного контекста:")
    print(f"  > num_nodes: {loaded_context['num_nodes']}")
    print(f"  > Ребра H_to_S: {loaded_context.get('edges_H_to_S').shape if loaded_context.get('edges_H_to_S') is not None else 'N/A'}")
    print(f"  > Произвольная маска H: {loaded_context.get('mask_H')}")
    print(f"  > Скрытые маппинги присутствуют: {'__RAW_GLOBAL_MAPPING__' in loaded_context}")

    
    # Проверка признаков и маппинга
    features_H = loaded_context.get('features_H')
    if features_H is not None:
        local_idx_9000_H = loaded_context['mapping']['H'].get(9000)
        if local_idx_9000_H is not None:
            print(f"  > Признаки 9000 в H (локальный индекс {local_idx_9000_H}): {features_H[local_idx_9000_H]}")
    
    # --- Очистка ---
    if os.path.exists(SAVE_FILE):
        os.remove(SAVE_FILE)