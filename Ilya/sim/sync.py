import pygame
from random import randrange
screensize = [1920, 1080]
sc = pygame.display.set_mode(screensize)
work = True
scale = 10
item_array = []
particle_array = []

Dark = pygame.Surface(screensize)
Dark.set_alpha(10)

B = 50*scale/2


class o():
    def __init__(self, I = randrange(-1, 2, 2), position = [screensize[0]//2, screensize[0]//2]) -> None:
        self.I = I
        self.pos = complex(*position)
        if self.I == 1:
            self.color = (255, 255, 0)
        elif self.I == -1:
            self.color = (0, 255, 255)
        else:
            self.color = (0, 0, 0)

    def update(self):
        for _ in range(0):
            particle_array.append(particle(self.pos + randrange(0, 10)*scale*1j**(randrange(0, 401)/100)))
        # pygame.draw.circle(sc, self.color, (self.pos.real, self.pos.imag), 5*abs(self.I))
dt = 0.01

class target_named_y:
    def __init__(self):
        self.pos = 0.0
        self.b = 0.0
        self.ba = 0.0
        self.air_k = 0.01**(dt)

    def epsillon(self, x):
        return (x - (self.pos + self.b)).real
    
    def update(self, x):
        E = self.epsillon(x)
        self.ba += kb*E * dt
        self.b += self.ba*dt
        self.ba *= self.air_k
        p = self.pos + self.b
        g = self.ba + p
        #pygame.draw.circle(sc, (255, 255, 0), (p.real, p.imag), 3)
        pygame.draw.line(sc, (255, 255, 0), (p.real, p.imag), (g.real, g.imag))
        return E

class particle():
    def __init__(self, pos, color = None, ls = 0.01, surf = sc) -> None:
        self.pos = pos
        self.movement = 0
        self.life = 1
        self.color = color
        self.ls = ls
        self.sc = surf

    def update(self, forces = 0.0):
        self.life -= self.ls
        e = self.pos
        D = 0
        Il = 0
        rm = 100000
        gm = 0
        for i in item_array:
            g = i.pos - self.pos
            r = abs(g)
            if rm > r:
                rm = r
                gm = g
                Il = i.I
            if r != 0:
                g /=r
                D += 1j*i.I*g/100
                D += i.I*complex(g.imag, -g.real)/50000
        #D = Il*complex(gm.imag, -gm.real)/5000
        E = abs(D*B)
        self.pos += (100*B*D + forces)*dt
        I = min(255, 255*E/10)
        if self.color is None:
            if E <= 70:
                pygame.draw.line(self.sc, (255 - I, 0, I), (self.pos.real, self.pos.imag), (e.real, e.imag))
        else:
            pygame.draw.line(self.sc, self.color, (self.pos.real, self.pos.imag), (e.real, e.imag), 2)



def draw_E():
    for xi in range(screensize[0]//scale):
        for yi in range(screensize[1]//scale):
            D = 0
            p = scale*complex(xi, yi)
            for i in item_array:
                g = i.pos - p
                r = abs(g)
                if r != 0:
                    g /=r
                    D += i.I*g
            p1 = p + B*D
            pygame.draw.line(sc, (255, 255, 0), (p.real, p.imag), (p1.real, p1.imag))

def I_update():
    for i in item_array:
        i.update()


def add_particle1(N):
    for n in range(N):
        pos = complex(randrange(0, screensize[0]), randrange(0, screensize[1]))
        for i in item_array:
            if abs(pos - i.pos) < scale*10:
                particle_array.append(particle(pos))
                break


def add_particle(N, surf = sc):
    for n in range(N):
        particle_array.append(particle(complex(*[randrange(0, s) for s in screensize]), surf = surf))


def P_update():
    for p in particle_array:
        if p.life > 0:
            p.update()
        else:
            particle_array.remove(p)    

k = 0

kx = 1
kb = 10.0
G = particle(complex(*screensize)/2 - 500, (255, 255, 255), ls = 0)
Q = particle(complex(*screensize)/2 + 500, (255, 255, 255), ls = 0)
target = target_named_y()
target1 = target_named_y()


def generate_PH_SPACE(N = 10, t = 200):
    PH = pygame.Surface(screensize)
    for _ in range(t):
        add_particle(N, PH)
        PH.blit(Dark, (0, 0))
        I_update()
        P_update()
    return PH

PH = generate_PH_SPACE()

def random_c():
    return complex(randrange(0, screensize[0]), randrange(0, screensize[1]))

while work:
    sc.blit(PH, (0, 0))
    sc.blit(Dark, (0, 0))
    target.pos = Q.pos#complex(*pygame.mouse.get_pos())
    target1.pos = G.pos
    # draw_E()
    eps = target.update(G.pos)
    eps1 = target.update(Q.pos)
    G.update(-kx*eps)
    Q.update(-kx*eps1)
    Err = (eps1**2 + eps**2)/100
    pygame.draw.circle(sc, (255, 0, 0), pygame.mouse.get_pos(), Err, 1)
#    pygame.time.Clock().tick(500)
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            work = False
            break
        if ev.type == pygame.MOUSEBUTTONDOWN:
            if ev.button == 1:
                item_array.append(o(1j**k, pygame.mouse.get_pos()))
            if ev.button == 3:
                item_array.append(o(-1j**k, pygame.mouse.get_pos()))

            if ev.button == 6:
                # item_array.append(o(1j**k, pygame.mouse.get_pos()))
                k += 0.5
            if ev.button == 7:
                k -= 0.5
                # item_array.append(o(-1j, pygame.mouse.get_pos()))
            PH = generate_PH_SPACE()


        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_r:
                G = particle(random_c(), (255, 255, 255), ls = 0)
                Q = particle(random_c(), (255, 255, 255), ls = 0)
                target = target_named_y()
            if ev.key == pygame.K_c:
                item_array.clear()
                G = particle(complex(*screensize)/2 - 500, (255, 255, 255), ls = 0)
                Q = particle(complex(*screensize)/2 + 500, (255, 255, 255), ls = 0)
                target = target_named_y()
                sc.fill([0]*3)


    pygame.display.update()