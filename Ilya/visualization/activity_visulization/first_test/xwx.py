from visualization import *
import numpy as np
path_to_scales = "./data/xwx/xwx_activity/activity.csv"


# просчитать позиции
def process(pushing_force=5.0, pulling_force=5.0, chunksize=100):
    path_to_activity = path_to_scales
    path_to_weights = "./data/xwx/xwx_activity/weights.csv"
    d = data(path_to_activity, path_to_weights, chunksize)
    d.process_and_merge_chunks(
        "./data/xwx/generated_positions.csv",
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
        "./data/xwx/generated_positions.csv",
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
        "./data/xwx/generated_positions.csv",
        update_axis_limits=True,
        path_to_scales=path_to_scales,
        max_size=10,
        min_size=1,
        save_path="./data/xwx/animation.mp4",
        ignore_scale_for_inds=ignore_neurons,
        #scale_activation_function=scale_activation_function,
    )


process(pushing_force=0.1, pulling_force=0.1, chunksize = 500) # просчитать позиции

a([]) # показать анимацию в окне

#b([]) # сохранить анимацию в файл


