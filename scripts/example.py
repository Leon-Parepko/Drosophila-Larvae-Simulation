#example.py
#OurHolyEnv/bin/python env_test/example.py
import arbor as arb
import data_preparation as dp
from arbor_recipes import basic_recipe
print('arb_config =', arb.config())

neurons_ids = ['7055857', '1805418', '14260575', '5835799', '10160250', '7840203', '5019924', '13986477', '10167078', '7982896', '4119387', '17591442', '4227544', '10495502', '8069478', '3913629', '11279244', '16846805', '8980589', '3664102']
neurons_ids = [int(i) for i in neurons_ids]

Sctx = dp.simulation_context('env_test/data', neurons_ids)
#Sctx.check_neurons()
# MPI CTX
arb.mpi_init()
comm = arb.mpi_comm()
print(comm)
ctx = arb.context(threads = 20, mpi=comm, gpu_id=0)
print(ctx)

onn = basic_recipe(Sctx)

decomp = arb.partition_load_balance(onn, ctx)


sim = arb.simulation(onn, ctx, decomp)
soma_handles = [sim.sample(gid, 'soma', arb.regular_schedule(0.1*arb.units.ms)) for gid in range(onn.num_cells())]

#simulation part
print("strted")
sim.run(tfinal=300 * arb.units.ms)
print(sim.samples(soma_handles[0]))
print('success')
'''
#getting samples
o = sim.samples(soma_handles[0])
t = o[0][0][:, 0]
U = o[0][0][:, 1]


#plotting results
import matplotlib.pyplot as plt
plt.plot(t, U)
plt.savefig("voltage_plot.png")
'''