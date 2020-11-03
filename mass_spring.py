"""
Explicit Solvers for static and dynamic spring systems
"""
from math import sin, cos, pi, sqrt
import numpy as np
import matplotlib.pyplot as plt
from spring_plotter import spring


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

        self._prev_vel = 0
        self._prev_mid_vel = 0
        self._mid_vel = 0
        self._vel = 0

        self._def = 0
        self._prev_def = 0

        self._displacement = []

    @property
    def deformation(self):
        """
        deformation as a numpy array
        """
        return np.asarray(self._displacement)

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
        self._def = init_disp
        self._vel = init_vel
        self._prev_def = init_disp
        self._prev_vel = init_vel

        acc = self._get_rhs(0) / self._get_lhs()
        self._prev_mid_vel = init_vel - self._dt / 2 * acc

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
        solves the dynamic system explicitly using second
        order central difference scheme

        Parameters
        ----------
        end_time : float
            end time to terminate the solver
        """
        time_array = [0]
        self._displacement.append(self._def)
        curr_time = self._dt
        while curr_time <= end_time:
            lhs = self._get_lhs()
            rhs = self._get_rhs(curr_time - self._dt)
            acceleration = rhs / lhs

            self._vel = self._prev_mid_vel + self._dt / 2 * acceleration
            self._mid_vel = self._prev_mid_vel + self._dt * acceleration
            self._def = self._prev_def + self._dt * self._mid_vel

            self._prev_mid_vel = self._mid_vel
            self._prev_def = self._def

            self._displacement.append(self._def)
            time_array.append(curr_time)
            curr_time += self._dt
        return np.array(time_array)


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
        int_force = self._get_internal_force(self._def)
        rhs = ext_force - int_force
        return rhs

    def solve_analytical(self, x0, v0):
        """
        get analytical solution of the system
        returns a function of t as an array or a list
        """
        eigen = sqrt(self._k / self._m)
        return lambda t: v0 / eigen * np.sin(eigen * t) + x0 * np.cos(eigen * t)


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
        rhs -= self._c * self._vel
        return rhs

    def solve_analytical(self, x0, v0):
        raise NotImplementedError("Too complicated :p")


if __name__ == "__main__":
    # == Static Spring
    SYSTEM = StaticSpring(5)
    SYSTEM.set_external_force(lambda t: 10)
    SYSTEM.solve()
    print(f"x = {SYSTEM.position[0]}")

    # == Undamped dynamic spring-mass system
    SYSTEM = MassSpring(m=1.0, k=1.0, dt=0.25)
    SYSTEM.set_external_force(external_force=0)
    SYSTEM.set_initial_conditions(init_disp=1.0, init_vel=-1.0)
    t = SYSTEM.solve(end_time=20)
    plt.figure()
    plt.plot(t, SYSTEM.deformation, "-o", label=r"$\Delta t=0.1$")
    plt.plot(t, SYSTEM.solve_analytical(1.0, -1.0)(t), "-", label="Analytical")
    TITLE = "Undamped spring mass system\n"
    TITLE += r"$M=1.0,~K=1.0,~F=0,~u_0=1.0,~\dot{u}_0=-1.0$"
    plt.title(TITLE)
    plt.xlabel("time")
    plt.ylabel("displacement")
    plt.legend()
    plt.grid()

    # == Damped dynamic spring-mass system
    SYSTEM = MassSpringDamper(m=1, c=0.03, k=lambda x: 1.0 * sin(x), dt=0.1)
    SYSTEM.set_external_force(lambda t: 3 * cos(4 * t))
    SYSTEM.set_initial_conditions(init_disp=1, init_vel=0)
    t = SYSTEM.solve(60)
    plt.figure()
    plt.plot(t, SYSTEM.deformation, label="0.1")
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("displacement")
    TITLE = "Damped spring mass system\n"
    TITLE += r"$M=1.0,~C=0.03,~K=\sin(u),~F=3\cos(t),~u_0=1.0,\dot{u}_0=0$"
    plt.title(TITLE)

    # fig, ax = plt.subplots()
    # (line,) = ax.plot([], [], c="black")
    # mass = ax.scatter([0], [0], s=750, c="black")
    # ax.set(xlim=[-1.5, 3.5], ylim=[-1.0, 1.0])
    # plt.axis("off")
    # for deform in SYSTEM.deformation:
    #     coords = spring([0, 0], [2 + deform, 0], 12, 0.5)
    #     line.set_xdata(coords[0])
    #     line.set_ydata(coords[1])
    #     mass.set_offsets([2.0 + deform, 0])
    #     plt.pause(0.00000001)
    # ax.set_aspect("equal", "box")
    plt.show()
