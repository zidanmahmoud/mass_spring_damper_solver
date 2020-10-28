"""
Explicit Solvers for static and dynamic spring systems
"""
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi


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

    def _get_external_force(self, time):
        if callable(self._f_ext):
            return self._f_ext(time)
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
        self._xm = 0
        self._xi = 0
        self._xp = 0

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
        self._xm = init_disp - self._dt * init_vel

    def _get_external_force(self, time):
        if callable(self._f_ext):
            return self._f_ext(time)
        return self._f_ext

    def _get_internal_force(self, disp):
        if callable(self._k):
            return self._k(disp)
        return self._k * disp

    def _get_lhs(self):
        raise NotImplementedError("Base Class!")

    def _get_rhs(self, time):
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
            disp = rhs / lhs
            self._xp = disp
            self._xm = self._xi
            self._xi = self._xp
            self.position.append(disp)


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
        return self._m / self._dt ** 2

    def _get_rhs(self, time):
        ext_force = self._get_external_force(time - self._dt)
        int_force = self._get_internal_force(self._xi)
        mass_term = self._m / self._dt ** 2
        rhs = ext_force - int_force + mass_term * (2 * self._xi - self._xm)
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

    def _get_rhs(self, time):
        ext_force = self._get_external_force(time - self._dt)
        int_force = self._get_internal_force(self._xi)
        mass_term = self._m / self._dt ** 2
        damp_term = self._c / (self._dt)
        rhs = (
            ext_force
            - int_force
            + mass_term * (2 * self._xi - self._xm)
            - damp_term * (self._xi - self._xm)
        )
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
