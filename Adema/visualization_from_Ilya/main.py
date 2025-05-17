from visualization import *

path_to_scales = "./data/echo_state_matrix.csv"


# просчитать позиции
def process(pushing_force=5.0, pulling_force=5.0, chunksize=100):
    path_to_activity = "./data/echo_state_matrix.csv"
    path_to_weights = "./data/adj.csv"
    d = data(path_to_activity, path_to_weights, chunksize)
    d.process_and_merge_chunks(
        "./data/generated_positions.csv",
        pushing_force=pushing_force,
        pulling_force=pulling_force,
    )


# показать анимацию в окне
def a():
    global path_to_scales
    visualize_positions(
        "./data/generated_positions.csv",
        update_axis_limits=True,
        
        max_size=100,
        min_size=0,
    )


# сохранить анимацию в файл
def b():
    global path_to_scales
    visualize_positions(
        "./data/generated_positions.csv",
        update_axis_limits=True,
        path_to_scales=path_to_scales,
        max_size=100,
        min_size=0,
        save_path="./data/animation.gif",
    )


process(pushing_force=5.0, pulling_force=5.0, chunksize = 100) # просчитать позиции
#a() # показать анимацию в окне
# b() # сохранить анимацию в файл
