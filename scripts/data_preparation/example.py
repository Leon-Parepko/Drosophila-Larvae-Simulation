import data_preparation as dp
neurons_ids = [29, 9469519]
ctx = dp.simulation_context('data/example', neurons_ids)
ctx.build_full_graph()