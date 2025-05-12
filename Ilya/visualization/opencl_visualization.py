import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf
import pandas as pd
import activity_generator as ag

class data:
    def __init__(self, data_path, chunk_size = 500):
        self.path = data_path
        self.chunk_size = chunk_size
        self.chunks_dataframe = pd.read_csv(self.path, chunksize=chunk_size)

    def update_chunk(self, chunk:pd.DataFrame):
        activity = chunk.to_numpy(np.float32)
        c = context(activity_dim=activity.shape[1])
        c.cdb(activity)


    def get_clasters(self, start_ind = 0, end_ind = -1):
        for n, chunk in enumerate(self.chunks_dataframe):
            if start_ind <= n < end_ind:
                self.update_chunk(chunk)
            elif n >= end_ind:
                break

class context:
    def __init__(self, activity_dim, cl_context = None):
        self.activity_dim = activity_dim
        if cl_context is None:
            cl_context = cl.create_some_context(0)
        self.ctx = cl_context
        self.queqe = cl.CommandQueue(self.ctx)
        self.program = cl.Program(self.ctx, self.source_code).build()

    def ccb(self, a:np.ndarray):
        return cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=a.astype(np.float32))

    def cdb(self, a: np.ndarray):
        return cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=a.astype(np.float32))

    @property
    def source_code(self):
        with open("./kernels.cl", 'r') as j:
            code = j.read()
        return code

    # кернел для поиска матрицы разности
    def get_difference_matrix(self, a:np.ndarray, b:np.ndarray):
        shape = len(a), len(b)
        self.program.difference_matrix(self.queqe, shape, None, )
    
    def simulate_clasters(self, activity, w, positions, dt):
        T = range(len(activity))
        ab = self.cdb(activity)
        w = self.ccb(w)
        positions = self.cdb(positions)
        next_positions = self.cdb(np.empty_like(positions))
        DT = self.ccb(np.array(dt))

        self.program.simulate_clasters(self.queqe, (self.activity_dim, ), None, )


data('./some_activity.csv', 3).get_clasters(end_ind = 2)