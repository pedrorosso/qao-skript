from numpy import array, cos, linspace, pi, sin, zeros
from matplotlib.pyplot import subplots

from src.materials import D_AlN, D_Sapphire, D_LiNb03
from src.christoffel import Christoffel

def surface_plotter(material: dict, n_theta: int = 1001, fname : str = 'test'):
    theta = linspace(0, 2 * pi, n_theta)  # azimuthal angle
    phi = array([0.0, 0.5 * pi])

    radius = zeros(shape=(3, 2, n_theta))
    qx = zeros(shape=(3, 2, n_theta))
    qy = zeros(shape=(3, 2, n_theta))
    qz = zeros(shape=(3, 2, n_theta))

    christoffel = Christoffel(stiffness=material['C'], density=material['rho'])

    for i, t in enumerate(theta):

        christoffel.set_direction_spherical(t, phi[0])
        radius[:, 0, i] = 1.0 / christoffel.get_phase_velocity()
        qx[:, 0, i] = radius[:, 0, i] * sin(t) * cos(phi[0])
        qy[:, 0, i] = radius[:, 0, i] * sin(t) * sin(phi[0])
        qz[:, 0, i] = radius[:, 0, i] * cos(t)

        christoffel.set_direction_spherical(t, phi[1])
        radius[:, 1, i] = 1.0 / christoffel.get_phase_velocity()
        qx[:, 1, i] = radius[:, 1, i] * sin(t) * cos(phi[1])
        qy[:, 1, i] = radius[:, 1, i] * sin(t) * sin(phi[1])
        qz[:, 1, i] = radius[:, 1, i] * cos(t)

    factor = 0.7
    size = (13.0 * factor, 6.0 * factor)
    fig, axs = subplots(ncols=2, nrows=1, facecolor='white', figsize=size)

    # kx - kz projections
    axs[0].plot(qx[0, 0, :],  qz[0, 0, :], color='C0', label='Quasi-Trans.')
    axs[0].plot(qx[1, 0, :],  qz[1, 0, :], color='C0')
    axs[0].plot(qx[2, 0, :],  qz[2, 0, :], color='C1', label='Quasi-Long.')
    axs[0].legend(loc='best')

    # ky - kz projections
    axs[1].plot(qy[0, 1, :],  qz[0, 1, :], color='C0', label='Quasi-Trans.')
    axs[1].plot(qy[1, 1, :],  qz[1, 1, :], color='C0')
    axs[1].plot(qy[2, 1, :],  qz[2, 1, :], color='C1', label='Quasi-Long.')
    axs[1].legend(loc='best')

    axs[0].set_xlabel(r'$q_{x} \ / \ \omega$ ', fontsize='large')
    axs[0].set_ylabel(r'$q_{z} \ / \ \omega$ ', fontsize='large')
    axs[1].set_xlabel(r'$q_{y} \ / \ \omega$ ', fontsize='large')
    axs[1].set_ylabel(r'$q_{z} \ / \ \omega$ ', fontsize='large')
    fig.tight_layout()

    fig.savefig(fname=f'{fname}.pdf')

    return qx[2, 0, :], qz[2, 0, :], qy[2, 1, :], qz[2, 1, :]


def plot_all_slowness_surfs():
    surface_plotter(material=D_LiNb03, n_theta=1000, fname='img/LiNBO3')
    surface_plotter(material=D_Sapphire, n_theta=1000, fname='img/Sapphire')
    surface_plotter(material=D_AlN, n_theta=1000, fname='img/AlN')

if __name__ == "__main__":
    plot_all_slowness_surfs()
