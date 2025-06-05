from math import e
from visualization import *
import numpy as np
path_to_scales = "./data/smooth/111/exp_3_sin/activity_matrix.csv"


# просчитать позиции
def process(pushing_force=5.0, pulling_force=5.0, chunksize=100):
    path_to_activity = path_to_scales
    path_to_weights = "./data/smooth/111/exp_3_sin/adj.csv"
    d = data(path_to_activity, path_to_weights, chunksize)
    d.process_and_merge_chunks(
        "./data/generated_positions.csv",
        pushing_force=pushing_force,
        pulling_force=pulling_force,
        dt = 0.1,
        end_ind = -1
    )


def scale_activation_function(x):
    return 1.0/(np.exp(-3*(x-2)) + 1)

# показать анимацию в окне
def a(ignore_neurons):
    global path_to_scales
    visualize_positions(
        "./data/generated_positions.csv",
        update_axis_limits=False,
        path_to_scales=path_to_scales,
        max_size=10,
        min_size=1,
        ignore_scale_for_inds=ignore_neurons,
        #scale_activation_function=scale_activation_function,
    )


# сохранить анимацию в файл
def b(ignore_neurons):
    global path_to_scales
    visualize_positions(
        "./data/generated_positions.csv",
        update_axis_limits=True,
        path_to_scales=path_to_scales,
        max_size=10,
        min_size=1,
        save_path="./data/animation.mp4",
        ignore_scale_for_inds=ignore_neurons,
        #scale_activation_function=scale_activation_function,
    )

ignore_neurons = [8980589]
def get_indexes_by_ids(ids:list[int]):
    output = []
    with open("./data/smooth/111/exp_3_sin/adj.csv", "r") as f:
        heads = f.readline()
        heads = heads.split(",")
        for n, i in enumerate(heads):
            if len(i) > 0:
                i = int(i)
            if i in ids:
                output.append(n)
    return output

ignore_neurons = get_indexes_by_ids([8980589])
#print(ignore_neurons)
#process(pushing_force=0.0, pulling_force=1.0, chunksize = 300) # просчитать позиции
#a([]) # показать анимацию в окне
b(ignore_neurons) # сохранить анимацию в файл