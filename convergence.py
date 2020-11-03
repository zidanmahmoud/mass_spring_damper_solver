"""
Testing second order convergence
"""

from math import sin, cos
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from mass_spring import MassSpring, MassSpringDamper


def plot_convergence_undamped():
    dts = np.geomspace(0.1, 0.0001, 4)
    errors = []
    system = MassSpring(m=1, k=1, dt=0.1)
    analytical = system.solve_analytical(-5, 10)
    for dt in dts:
        print(f"dt={dt}")
        system = MassSpring(m=1, k=1, dt=dt)
        system.set_initial_conditions(init_disp=-5, init_vel=10)
        t = system.solve(end_time=100)
        y = analytical(t)
        errors.append(np.linalg.norm(y - system.deformation) / np.linalg.norm(y) * 100)
    _convegence_plot(dts, errors, "Convergence for the undamped mass-spring system")


def plot_convergence_damped_linear():
    dts = np.geomspace(0.1, 1e-5, 5)
    errors = []

    system_fine = MassSpringDamper(m=1, c=0.1, k=1.0, dt=1e-6)
    system_fine.set_initial_conditions(init_disp=1, init_vel=0)
    t_fine = system_fine.solve(60)
    def_fine = system_fine.deformation
    for dt in dts:
        print(f"dt={dt}")
        system = MassSpringDamper(m=1, c=0.1, k=1.0, dt=dt)
        system.set_initial_conditions(init_disp=1, init_vel=0)
        t = system.solve(60)
        solution = system.deformation
        analytical = np.interp(t, t_fine, def_fine)
        errors.append(
            np.linalg.norm(analytical - solution) / np.linalg.norm(analytical) * 100
        )
    _convegence_plot(
        dts, errors, "Convergence for the linear damped mass-spring system"
    )


def plot_convergence_damped_nonlinear():
    dts = np.geomspace(0.1, 1e-5, 5)
    errors = []

    system_fine = MassSpringDamper(m=1, c=0.03, k=lambda x: 1.0 * sin(x), dt=1e-6)
    system_fine.set_external_force(lambda t: 3 * cos(4 * t))
    system_fine.set_initial_conditions(init_disp=1, init_vel=0)
    t_fine = system_fine.solve(60)
    def_fine = system_fine.deformation
    for dt in dts:
        print(f"dt={dt}")
        system = MassSpringDamper(m=1, c=0.03, k=lambda x: 1.0 * sin(x), dt=dt)
        system.set_external_force(lambda t: 3 * cos(4 * t))
        system.set_initial_conditions(init_disp=1, init_vel=0)
        t = system.solve(60)
        solution = system.deformation
        analytical = np.interp(t, t_fine, def_fine)
        errors.append(
            np.linalg.norm(analytical - solution) / np.linalg.norm(analytical) * 100
        )
    _convegence_plot(
        dts, errors, "Convergence for the non-linear damped mass-spring system"
    )


def _convegence_plot(dts, errors, plt_title):
    _, ax = plt.subplots()
    ax.plot(dts, errors, "o-")
    dts = np.log(dts)
    errors = np.log(errors)
    regression = linregress(dts, errors)
    print(f"order of convergence: {regression.slope}")
    plt.grid()
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"$\Delta t$",
        ylabel="error [%]",
        title=plt_title,
    )
    ax.annotate(
        f"order = {regression.slope:.3f}",
        xy=(1, 0),
        xycoords="axes fraction",
        fontsize=16,
        xytext=(-5, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
    )


def show_plots():
    plt.show()


if __name__ == "__main__":
    plot_convergence_undamped()
    # plot_convergence_damped_linear()
    plot_convergence_damped_nonlinear()
    show_plots()
