"""
Testing second order convergence
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from mass_spring import MassSpring


if __name__ == "__main__":
    dts = np.geomspace(0.1, 0.00001, 5)
    errors = []
    for dt in dts:
        SYSTEM = MassSpring(m=1, k=1, dt=dt)
        SYSTEM.set_external_force(external_force=0)
        SYSTEM.set_initial_conditions(init_disp=-5, init_vel=10)
        t = SYSTEM.solve_central(end_time=10)
        x, y = SYSTEM.solve_analytical(-5, 10, end_time=10)
        errors.append(np.linalg.norm(y - SYSTEM.deformation) / np.linalg.norm(y) * 100)

    fig, ax = plt.subplots()
    ax.plot(dts, errors, "o-")
    dts = np.log(dts)
    errors = np.log(errors)
    print(f"order of convergence: {linregress(dts, errors).slope}")
    plt.grid()
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"$\Delta t$",
        ylabel="error [%]",
        title="Second order convergence for undamped mass-spring system",
    )
    plt.show()
