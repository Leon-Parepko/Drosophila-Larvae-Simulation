import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf
import pandas as pd
import activity_generator as ag

import tempfile
import os

class data:
    def __init__(self, data_path, w_path:str, chunk_size = 500): 
        self.path = data_path
        self.w_path = w_path
        self.chunk_size = chunk_size
        self.chunks_dataframe = pd.read_csv(self.path, chunksize=chunk_size)
        self.w = pd.read_csv(self.w_path).to_numpy(np.float32)

    def update_chunk(self, chunk:pd.DataFrame, ctx:'context', last_positions, dt = 0.01, save_positions_every = 1):
        activity = chunk.to_numpy(np.float32)
        positions = ctx.simulate_clasters(activity, self.w, last_positions, dt, save_positions_every)
        return positions

    def get_clasters(self, output_chunks_directory:str, start_ind=0, end_ind=-1, last_positions=None, dt=0.01, save_positions_every=1):
        activity_dim = self.w.shape[0]
        c = context(activity_dim=activity_dim)
        if last_positions is None:
            last_positions = np.random.uniform(-2, 2, (activity_dim * 2)).astype(np.float32)
        for n, chunk in enumerate(self.chunks_dataframe):
            if start_ind <= n and (n < end_ind or end_ind == -1):
                positions = self.update_chunk(chunk, c, last_positions, dt=dt, save_positions_every=save_positions_every)
                last_positions = positions[-1]
                print('chunk', n, 'generated')

                name = f'chunk{n} shape_{activity_dim}x2 every_{save_positions_every}_tick.csv'
                pd.DataFrame(positions, columns=[i for i in range(activity_dim*2)]).to_csv(pp := os.path.join(output_chunks_directory, name))

                print('chunk', n, 'saved to', pp)
            elif n >= end_ind:
                break
        return positions
    
    def save_clusters(self, name, positions):
        pd.DataFrame(positions).to_csv(name)

class context:
    def __init__(self, activity_dim, cl_context = None):
        self.activity_dim = activity_dim
        if cl_context is None:
            cl_context = cl.create_some_context(0)
        self.ctx = cl_context
        self.queqe = cl.CommandQueue(self.ctx)
        self.program = cl.Program(self.ctx, self.source_code).build()

    def ccb(self, a:np.ndarray, dtype = np.float32):
        return cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=a.astype(dtype))

    def cdb(self, a: np.ndarray, dtype=np.float32):
        return cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=a.astype(dtype))

    @property
    def source_code(self):
        with open("./kernels.cl", 'r') as j:
            code = j.read()
        return code

    # кернел для поиска матрицы разности
    def get_difference_matrix(self, a:np.ndarray, b:np.ndarray):
        pass
        #shape = len(a), len(b)
        #self.program.difference_matrix(self.queqe, shape, None, )

    def simulate_clasters(self, activity, w, positions, dt, save_positions_every = 1):
        output_positions = []
        flattened_positions = np.reshape(positions, positions.shape[0]*positions.shape[1]) if len(positions.shape) > 1 else positions

        ab = self.cdb(np.reshape(activity, activity.shape[0]*activity.shape[1]))
        w = self.ccb(np.reshape(w, w.shape[0]*w.shape[1]))
        positions_ = self.cdb(flattened_positions)
        next_positions_ = self.cdb(np.empty_like(flattened_positions))
        dt_ = self.ccb(np.array(dt))
        activity_size_ = self.ccb(np.array(activity.shape), np.int32)

        for ti in range(len(activity)):
            t = self.ccb(np.array(ti), np.int32)
            self.program.simulate_clasters(self.queqe, (self.activity_dim, ), None, ab, w, positions_, next_positions_, activity_size_, t, dt_)
            positions_, next_positions_ = next_positions_, positions_
            if ti % save_positions_every == 0:
                op = np.empty_like(flattened_positions)
                cl.enqueue_copy(self.queqe, op, positions_)
                output_positions.append(op)
        
        return output_positions

d = './some_acitivity'
path_to_activity = os.path.join(d, 'activity.csv')
path_to_weights = os.path.join(d, 'weights.csv')

if False: # генерируем файл с произвольной активностью (тут нет gpu по этому это медленный процесс)
    df_w, df = ag.generate_activity_df(ag.some_activity_system(group_complexity=30), 1000)
    print('activity generated')
    df.to_csv(path_to_activity)
    df_w.to_csv(path_to_weights)
    print('activity saved')

d = data(path_to_activity, path_to_weights, 100)
clasters = d.get_clasters('./output_chunks')