import pygame as pg
import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf


class cl_ctx:
    def __init__(self, sizes) -> None:
        self.ctx = cl.create_some_context(0)
        self.queqe = cl.CommandQueue(self.ctx)

        self.sizes = np.array(sizes)

        self.Ugrid = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=0.0*np.ones((sizes[0]*sizes[1]), np.float32))
        self.Ugrid = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=0.0*np.ones((sizes[0]*sizes[1]), np.float32))
        # self.current_wawe = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf = np.array(100.1*(np.random.sample((universe_size[0]*universe_size[1]*2))*0.5 - 0.25), np.float32))
        self.next_wawe = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=np.zeros(
            (sizes[0]*sizes[1]*2), np.float32))
        self.U = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=np.zeros(
            (sizes[0]*sizes[1]), np.float32))

        self.sizes_b = cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(
            self.sizes, np.int32))

        with open("kernels.cl", 'r') as j:
            self.program = cl.Program(self.ctx, j.read()).build()

        self.movement_update = self.program.movement_update
        self.Posd_update = self.program.Posd_update
        self._get_image = self.program.get_image
        self._draw_wawe = self.program.draw_wawe
        self._clear = self.program.clear
        self._draw_particle = self.program.draw_particle

        self.image_buff = cl.Buffer(self.ctx, mf.COPY_HOST_PTR, hostbuf=np.zeros(
            (sizes[0]*sizes[1]*3), np.int32))

    def update(self):
        self.movement_update(self.queqe, self.sizes,
                             None, self.Ugrid, self.movement, self.sizes_b)
        self.Posd_update(self.queqe, self.sizes, None,
                         self.Ugrid, self.next_wawe, self.movement, self.sizes_b)
        self.Ugrid, self.next_wawe = self.next_wawe, self.Ugrid

    def draw_wawe(self, position, radius, value):
        self._draw_wawe(self.queqe, (2*radius, 2*radius), None, self.Ugrid, self.movement, self.sizes_b,
                        cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(
                            position, np.int32) - np.array(radius, np.int32)),
                        cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY,
                                  hostbuf=np.array(radius, np.float32)),
                        cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(value, np.float32)))

    def draw_particle(self, position, radius, value):
        self._draw_particle(self.queqe, (2*radius, 2*radius), None, self.Ugrid, self.movement, self.sizes_b,
                            cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(
                                position, np.int32) - np.array(radius, np.int32)),
                            cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY,
                                      hostbuf=np.array(radius, np.float32)),
                            cl.Buffer(self.ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=np.array(value, np.float32)))

    def clear(self):
        self._clear(self.queqe, self.sizes, None,
                    self.Ugrid, self.next_wawe, self.movement, self.sizes_b)

    def get_image(self):
        self._get_image(self.queqe, self.sizes, None,
                        self.Ugrid, self.movement, self.image_buff, self.sizes_b)
        im = np.zeros(
            (self.sizes[0]*self.sizes[1]*3), np.int32)

        cl.enqueue_copy(self.queqe, im, self.image_buff)
        im.resize((self.sizes[0], self.sizes[1], 3))
        return im


screensize = np.array((1920, 1080))

sc = pg.display.set_mode(screensize)
Clock = pg.time.Clock()
work = True
s = 1
p = cl_ctx((1920//s, 1080//s))

time = 0

pause = 1
while work:
    time += 0.05
    im = p.get_image()
    im = pg.pixelcopy.make_surface(im)
    im = pg.transform.scale(im, screensize)
    sc.blit(im, (0, 0))
    # Clock.tick(30)
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
        p.draw_particle(p.sizes * np.array(pg.mouse.get_pos()) /
                        screensize, 20, [5, 0])
    if mkey[0]:
        p.draw_wawe(p.sizes * np.array(pg.mouse.get_pos()) /
                    screensize, 10, [5, 0])
    if (pause == 1):
        p.update()
    pg.display.update()
