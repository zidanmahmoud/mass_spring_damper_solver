"""
Explicit Solvers for static and dynamic spring systems
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi
from spring import spring


class StaticSpring:
    """
    Static spring solver

    Parameters
    ----------
    k : float
        spring stiffness
    """

    def __init__(self, k):
        self._k = float(k)
        self._f_ext = 0

        self.position = []

    def set_external_force(self, external_force):
        """
        sets the external force on the spring

        Parameters
        ----------
        external_force : callable or float
        """
        if callable(external_force):
            self._f_ext = external_force
        else:
            self._f_ext = float(external_force)

    def _get_external_force(self, time_):
        if callable(self._f_ext):
            return self._f_ext(time_)
        return self._f_ext

    def _get_internal_force(self, displacement):
        return self._k * displacement

    def solve(self):
        """
        solves the static system
        """
        int_force = self._get_internal_force(0)
        ext_force = self._get_external_force(0)
        rhs = ext_force - int_force
        lhs = self._k
        disp = rhs / lhs
        self.position.append(disp)


class DynamicSpring:
    """
    Base class for dynamic mass-spring explicit solver

    Parameters
    ----------
    m : float
        mass
    k : callable or float
        spring stiffness. callable for non-linear springs
    dt : float
        time step-size
    """

    def __init__(self, m, k, dt):
        self._m = float(m)
        if callable(k):
            self._k = k
        else:
            self._k = float(k)
        self._dt = float(dt)

        self._f_ext = 0
        self._vm = 0
        self._vp = 0
        self._xi = 0

        self.position = []

    def set_external_force(self, external_force):
        """
        sets the external force on the spring

        Parameters
        ----------
        external_force : callable or float
        """
        if callable(external_force):
            self._f_ext = external_force
        else:
            self._f_ext = float(external_force)

    def set_initial_conditions(self, init_disp, init_vel):
        """
        sets the initial displacement and velocity.
        The velocity is set using forward differencing scheme.

        Parameters
        ----------
        init_disp : float

        init_vel : float
        """
        self._xi = init_disp
        self._vm = init_vel

    def _get_external_force(self, time_):
        if callable(self._f_ext):
            return self._f_ext(time_)
        return self._f_ext

    def _get_internal_force(self, disp):
        if callable(self._k):
            return self._k(disp)
        return self._k * disp

    def _get_lhs(self):
        raise NotImplementedError("Base Class!")

    def _get_rhs(self, time_):
        raise NotImplementedError("Base Class!")

    def solve(self, end_time):
        """
        solves the dynamic system explicitly

        Parameters
        ----------
        end_time : float
            end time to terminate the solver
        """
        self.position.append(self._xi)
        curr_time = 0
        while curr_time <= end_time:
            curr_time += self._dt
            lhs = self._get_lhs()
            rhs = self._get_rhs(curr_time - self._dt)
            acceleration = rhs / lhs
            self._vp = self._vm + self._dt * acceleration
            self._xi += self._dt * self._vp
            self.position.append(self._xi)
            self._vm = self._vp


class MassSpring(DynamicSpring):
    """
    Dynamic mass-spring explicit solver

    Parameters
    ----------
    m : float
        mass
    k : callable or float
        spring stiffness. callable for non-linear springs
    dt : float
        time step-size
    """

    def _get_lhs(self):
        return self._m

    def _get_rhs(self, time_):
        ext_force = self._get_external_force(time_ - self._dt)
        int_force = self._get_internal_force(self._xi)
        rhs = ext_force - int_force
        return rhs


class MassSpringDamper(MassSpring):
    """
    Dynamic mass-damper-spring explicit solver

    Parameters
    ----------
    m : float
        mass
    c : float
        damping coefficient
    k : callable or float
        spring stiffness. callable for non-linear springs
    dt : float
        time step-size
    """

    def __init__(self, m, c, k, dt):
        super().__init__(m, k, dt)
        self._c = float(c)

    def _get_rhs(self, time_):
        rhs = super()._get_rhs(time_)
        rhs -= self._c * self._vm
        return rhs


if __name__ == "__main__":
    # == Static Spring
    SYSTEM = StaticSpring(5)
    SYSTEM.set_external_force(lambda t: 10)
    SYSTEM.solve()
    print(f"x = {SYSTEM.position[0]}")

    # == Undamped dynamic spring-mass system
    SYSTEM = MassSpring(1, 1, 0.05)
    SYSTEM.set_external_force(0)
    SYSTEM.set_initial_conditions(-5, 10)
    SYSTEM.solve(50)
    plt.figure()
    plt.plot(np.linspace(0, 10, num=len(SYSTEM.position)), SYSTEM.position)
    plt.grid()

    # == Damped dynamic spring-mass system
    SYSTEM = MassSpringDamper(1, 0.03, lambda x: 1 * sin(x), 0.05)
    SYSTEM.set_external_force(lambda t: 3 * cos(4 * t))
    SYSTEM.set_initial_conditions(1, 0)
    SYSTEM.solve(60)
    plt.figure()
    plt.plot(np.linspace(0, 60, num=len(SYSTEM.position)), SYSTEM.position)
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("displacement")
    TITLE = "Dynamic Damped spring mass system\n"
    TITLE += r"$M=1,~C=0.03,~K=\sin(x),~F=3\cos(x),~x_0=1,v_0=0$"
    plt.title(TITLE)

    plt.show()

    fig, ax = plt.subplots()
    (line,) = ax.plot([], [], c="black")
    mass = ax.scatter([0], [0], s=750, c="black")
    ax.set(xlim=[-1.5, 2.5], ylim=[-1.0, 1.0])
    plt.axis("off")
    for deformation in SYSTEM.position:
        coords = spring([0, 0], [1 + deformation, 0], 12, 0.5)
        line.set_xdata(coords[0])
        line.set_ydata(coords[1])
        mass.set_offsets([1.0 + deformation, 0])
        plt.pause(0.00000001)
    ax.set_aspect("equal", "box")
    plt.show()
