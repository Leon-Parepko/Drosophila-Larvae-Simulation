import pygame as pg
import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf


class cl_ctx:
    def __init__(self, sizes) -> None:
        self.ctx = cl.create_some_context(0)
        self.queqe = cl.CommandQueue(self.ctx)
        self.sizes = np.array(sizes)

        def create_grid_buff(value = .0): 
            return cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=value*np.ones((sizes[0]*sizes[1]), np.float32))
        
        self.Vgrid, self.Vgrid_next = create_grid_buff(0.5278033), create_grid_buff()
        self.mgrid, self.mgrid_next = create_grid_buff(0.05632488), create_grid_buff()
        self.hgrid, self.hgrid_next = create_grid_buff(0.5829663), create_grid_buff()
        self.ngrid, self.ngrid_next = create_grid_buff(0.31968778), create_grid_buff()
        self.change_grid = create_grid_buff(1.0)
        self.sizes_b = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(self.sizes, np.int32))

        with open("Ilya/sim/kernel.cl", 'r') as j:
            self.program = cl.Program(self.ctx, j.read()).build()

        self.__update_params = self.program.update_params
        self.__clear = self.program.clear
        self.__get_image = self.program.get_image
        self.__add_I = self.program.add_I
        self.__set_change_coof = self.program.set_change_coof

        self.image_buff = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=np.zeros((sizes[0]*sizes[1]*3), np.int32))

    def read(self):
        sizes = self.sizes
        v = np.empty((sizes[0]*sizes[1]), np.float32)
        m = np.empty((sizes[0]*sizes[1]), np.float32)
        h = np.empty((sizes[0]*sizes[1]), np.float32)
        n = np.empty((sizes[0]*sizes[1]), np.float32)
        cl.enqueue_copy(self.queqe, v, self.Vgrid)
        cl.enqueue_copy(self.queqe, m, self.mgrid)
        cl.enqueue_copy(self.queqe, h, self.hgrid)
        cl.enqueue_copy(self.queqe, n, self.ngrid)
        v.resize((self.sizes[0], self.sizes[1], 3))
        m.resize((self.sizes[0], self.sizes[1], 3))
        h.resize((self.sizes[0], self.sizes[1], 3))
        n.resize((self.sizes[0], self.sizes[1], 3))
        return v, m, h, n

    def update(self):
        self.__update_params(self.queqe, self.sizes, None,
                           self.Vgrid, self.mgrid, self.hgrid, self.ngrid, 
                           self.Vgrid_next, self.mgrid_next, self.hgrid_next, self.ngrid_next,
                           self.change_grid, self.sizes_b)

        self.Vgrid, self.Vgrid_next = self.Vgrid_next, self.Vgrid
        self.mgrid, self.mgrid_next = self.mgrid_next, self.mgrid
        self.hgrid, self.hgrid_next = self.hgrid_next, self.hgrid
        self.ngrid, self.ngrid_next = self.ngrid_next, self.ngrid

    def add_I(self, position, radius, value):
        self.__add_I(self.queqe, (2*radius, 2*radius), None, 
                            self.Vgrid, self.change_grid, self.sizes_b,
                            cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(position, np.int32) - np.array(radius, np.int32)),
                            cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(radius, np.float32)),
                            cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(value, np.float32)))

    def set_change_coof(self, position, radius, value = 1.0):
        self.__set_change_coof(self.queqe, (int(2*radius), int(2*radius)), None, 
                            self.change_grid, self.sizes_b,
                            cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(position, np.int32) - np.array(radius, np.int32)),
                            cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(radius, np.float32)),
                            cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(value, np.float32)))

    def clear(self):
        vV = 0.5278033
        vm = 0.05632488
        vh = 0.5829663
        vn = 0.31968778
        self.__clear(self.queqe, self.sizes, None, self.Vgrid, cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(vV, np.float32)), self.sizes_b)
        self.__clear(self.queqe, self.sizes, None, self.mgrid, cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(vm, np.float32)), self.sizes_b)
        self.__clear(self.queqe, self.sizes, None, self.hgrid, cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(vh, np.float32)), self.sizes_b)
        self.__clear(self.queqe, self.sizes, None, self.ngrid, cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(vn, np.float32)), self.sizes_b)
        self.__clear(self.queqe, self.sizes, None, self.change_grid, cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(0.0, np.float32)), self.sizes_b)
    
    def get_image(self):
        self.__get_image(self.queqe, self.sizes, None,
                        self.Vgrid, self.mgrid, self.hgrid, self.ngrid,
                        self.change_grid, self.image_buff, self.sizes_b)
        im = np.zeros((self.sizes[0]*self.sizes[1]*3), np.int32)
        cl.enqueue_copy(self.queqe, im, self.image_buff)
        im.resize((self.sizes[0], self.sizes[1], 3))
        return im

screensize = np.array((1920, 1080))

sc = pg.display.set_mode(screensize)
Clock = pg.time.Clock()
work = True
s = 10
p = cl_ctx((1920//s, 1080//s))

time = 0

pause = 1
while work:
    im = p.get_image()
    im = pg.pixelcopy.make_surface(im)
    im = pg.transform.scale(im, screensize)
    sc.blit(im, (0, 0))
    for ev in pg.event.get():
        if ev.type == pg.QUIT:
            work = False
            break
        if ev.type == pg.KEYDOWN:
            if ev.key == pg.K_SPACE:
                pause *= -1
            if ev.key == pg.K_c:
                p.clear()
    
    mkey = pg.mouse.get_pressed()
    if mkey[2]:
        p.set_change_coof(p.sizes * np.array(pg.mouse.get_pos())/screensize, 1.0, 1.0)
    if mkey[0]:
        p.add_I(p.sizes * np.array(pg.mouse.get_pos())/screensize, 2, 9.99)
    
    if (pause > 0):
        for _ in range(20):
            p.update()
    
    pg.display.update()

t = p.read()
print(*((n, q.max()) for n, q in zip("v m h n".split(' '), t)), sep = '\n')