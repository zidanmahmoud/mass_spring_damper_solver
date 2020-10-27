import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# class MassSpringSystem:
#     def __init__(self, mass, damping, stiffness):
#         self._m = mass
#         self._c = damping
#         self._k = stiffness

#         self._u_np1 = 0
#         self._u_n = 1
#         self._u_nm1 = 0

#     def _get_external_force(self, t):
#         return 0 * np.sin(0*t)

#     def _get_internal_force(self):
#         return -1.0 * self._k * (self._u_n)

#     # def solve_implicit(self, tf, dt):
#     #     t = dt
#     #     while t <= tf:
#     #         print(f"time = {t}")
#     #         # solution
#     #         v = (self._u_np1 - self._u_nm1) / (2 * dt)
#     #         a = (self._u_np1 - 2 * self._u_n + self._u_nm1) / (dt**2)
#     #         g = self._get_external_force(t)
#     #         f = self._get_internal_force()
#     #         rhs = g - f - (2*self._m/dt**2)*self._u_n - (self._m/dt**2 - self._c/(2*dt))*self._u_nm1
#     #         lhs = self._m/dt**2 + self._c/(2*dt)
#     #         self._u_np1 = rhs / lhs
#     #         print(f"current displacement = {self._u_np1}\n")

#     #         self._u_nm1 = self._u_n
#     #         self._u_n = self._u_np1
#     #         t += dt

#     def solve_explicit(self, tf, dt):
#         us = [self._u_n]
#         ts = [0]
#         t = dt

#         self._initialize_solver(dt)

#         while t <= tf:
#             print(f"time = {t}")
#             # solution

#             g = self._get_external_force(t)
#             f = self._get_internal_force()
#             rhs = g - f - self._A*self._u_nm1 + self._B*self._u_n
#             self._u_np1 = rhs / self._K
#             print(f"current displacement = {self._u_np1}\n")

#             self._u_nm1 = self._u_n
#             self._u_n = self._u_np1
#             t += dt
#             us.append(self._u_n)
#             ts.append(t)

#         return ts, us

#     def _initialize_solver(self, dt):
#         self._K = (self._m) / (dt**2) + (self._c) / (2*dt)
#         self._A = (self._m) / (dt**2) - (self._c) / (2*dt)
#         self._B = (2 * self._m) / (dt**2)


# if __name__ == "__main__":
#     system = MassSpringSystem(1, 0, 1)
#     t, u = system.solve_explicit(1, 0.01)
#     plt.plot(t, u)
#     plt.show()



class StaticSpring:
    def __init__(self, k):
        self._k = k
        self._f_ext = lambda t: 0

        self.position = []

    def set_external_force(self, external_force_callable):
        self._f_ext = external_force_callable

    def _get_external_force(self, t):
        return self._f_ext(t)

    def _get_internal_force(self, x):
        return self._k * x

    def solve(self):
        f = self._get_internal_force(0)
        g = self._get_external_force(0)
        RHS = g - f
        LHS = self._k
        x = RHS / LHS
        self.position.append(x)


class MassSpring:
    def __init__(self, m, k):
        self._m = m
        self._k = k
        self._f_ext = lambda t: 0

        self._xm = 0
        self._xi = 0
        self._xp = 0

        self.position = []

    def set_external_force(self, external_force_callable):
        self._f_ext = external_force_callable

    def set_initial_position(self, x0):
        self._xi = x0

    def _get_external_force(self, t):
        return self._f_ext(t)

    def _get_internal_force(self, x):
        return self._k * x

    def solve(self, tf, dt):
        self.position.append(self._xi)
        t = 0
        while t <= tf:
            t += dt
            g = self._get_external_force(t)
            f = self._get_internal_force(self._xi)

            M = self._m/dt**2
            RHS = g - f + M * 2 * self._xi - M * self._xm
            LHS = M
            x = RHS / LHS
            self._xp = x

            self._xi = self._xp
            self._xm = self._xi

            self.position.append(x)



if __name__ == "__main__":
    system = StaticSpring(5)
    system.set_external_force(lambda t: 10)
    system.solve()
    print(system.position)

    system = MassSpring(1, 5)
    system.set_external_force(lambda t: 10*np.sin(3*t))
    system.set_initial_position(1)
    system.solve(10, 0.1)
    plt.plot(system.position)
    plt.show()


    m = 1
    k = 5
    g = lambda t: 10*np.sin(3*t)
    pos = [0]
    xm = 0
    xp = 0
    xi = 0
    t = 0
    dt = 0.1
    M = m / (dt**2)
    while t<=10:
        t += dt
        xp = ( g(t) - k*xi + 2*M*xi - M*xm ) / M
        pos.append(xp)
        xi = xp
        xm = xi
    plt.plot(pos)
    plt.show()






#############################################################################

# M = 1
# C = 0
# K = 1
# def g(t):
#     return 0
# def f():
#     return - K * u

# u = 1
# um = 0
# up = 0

# tf = 50
# dt = 0.1

# K_star = M/(dt**2) + C/(2*dt)
# A = M/(dt**2) - C/(2*dt)
# B = 2*M/(dt**2)

# v0 = 0
# a0 = (g(0) - C*v0 - K*u) / M
# um = u - dt*v0 + (dt*2)/2 * a0

# us = [u]
# ts = [0]

# t = 0
# while t <= tf:
#     t += dt

#     rhs = g(t) + f() - A*um + B*u
#     up = rhs / K_star

#     us.append(up)
#     ts.append(t)

#     u = up
#     um = u

# plt.plot(ts, us)
# plt.show()

