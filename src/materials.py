from itertools import product as itertools_product
from typing import Callable, Optional, Tuple, Union

from matplotlib.pyplot import Figure, subplots
from numpy import extract, isclose, ndarray, array, linspace, zeros, pi, cos, sin, zeros_like, nditer
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator, splrep, splev

if __package__ is None or __package__ == '':
    import christoffel as ch
else:
    from . import christoffel as ch

"""
    Most of the stiffness tensors come from the Handbook of Constants HOC
    (https://link.springer.com/referencework/10.1007/3-540-30437-1)
    unless stated otherwise.
    
    For a good summary relating stiffness tensors and symmetries I recommend Cleland's 
    Book (https://link.springer.com/book/10.1007/978-3-662-05287-7)
"""

available_materials = ['D_Sapphire', 'D_GaAs', 'D_Quartz', 'D_Glass', 'D_TiO2', 'D_AlN', 'D_CaF2', 'D_LiNb03']

# Sapphire (alpha Aluminium Oxide, Al2O3)
# HOC 844
C_Sapphire = array([[496.0, 159.0, 114.0, -23.0, 0.0, 0.0],
                    [159.0, 496.0, 114.0, 23.0, 0.0, 0.0],
                    [114.0, 114.0, 499.0, 0.0, 0.0, 0.0],
                    [-23.0, 23.0, 0.0, 146.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 146.0, -23.0],
                    [0.0, 0.0, 0.0, 0.0, -23.0, 0.5 * (496.0 - 159.0)]])

D_Sapphire = {
    'name': r'$\alpha - Al_2 O_3$',
    'xy_length': 1000e-6,  # 0.5*1500.0e-6,
    'z_length': 0.41e-3,
    'grid_pts': 2 ** 7,
    'z_grid_pts': 101,
    'f_0': 6.29e9,  # phonon frequency
    'kappa': 0.0,
    'absorbRadius': 300e-6,
    'C': C_Sapphire,
    'rho': 3980.0
}

# Gallium Arsenide
# HOC p. 621
C_GaAs = array([[119.0, 53.8, 53.8, 0.0, 0.0, 0.0],
                [53.8, 119.0, 53.8, 0.0, 0.0, 0.0],
                [53.8, 53.8, 119.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 59.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 59.5, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 59.5]])

C_GaAs_no_ref = array([[99.0, 41.0, 41.0, 0.0, 0.0, 0.0],
                       [41.0, 99.0, 41.0, 0.0, 0.0, 0.0],
                       [41.0, 41.0, 99.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 51.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 51.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 51.0]])

D_GaAs = {
    'name': r'$GaAs$',
    'xy_length': 600e-6,  # 0.5*1500.0e-6,
    'z_length': 5e-3,
    'grid_pts': 2 ** 7,
    'z_grid_pts': 101,
    'f_0': 6.29e9,  # phonon frequency
    'kappa': 0.0,
    'absorbRadius': 250e-6,
    'C': C_GaAs,
    'rho': 5317.6
}

# Quartz (alpha Silicon Oxide)
# HOC 846
C_Quartz = array([[86.6, 6.7, 12.4, 17.8, 0.0, 0.0],
                  [6.7, 86.6, 12.4, -17.8, 0.0, 0.0],
                  [12.4, 12.4, 106.4, 0.0, 0.0, 0.0],
                  [17.8, -17.8, 0.0, 58.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 58.0, 17.8],
                  [0.0, 0.0, 0.0, 0.0, 17.8, 0.5 * (86.6 - 6.7)]])

D_Quartz = {
    'name': r'$\alpha - Si O_2$',
    'xy_length': 600e-6,  # 0.5*1500.0e-6,
    'z_length': 5e-3,
    'grid_pts': 2 ** 7,
    'z_grid_pts': 101,
    'f_0': 12.645e9,
    'kappa': 0.0,
    'absorbRadius': 250e-6,
    'C': C_Quartz,
    'rho': 2648.5
}

# BK7 Schott Glass
# HOC 828,
C_Glass = array([[92.3, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 92.3, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 92.3, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 92.3 / 2, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 92.3 / 2, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 92.3 / 2]])

D_Glass = {
    'name': 'BK7',
    'xy_length': 600e-6,  # 0.5*1500.0e-6,
    'z_length': 5e-3,
    'grid_pts': 2 ** 7,
    'z_grid_pts': 101,
    'f_0': 12.645e9,
    'kappa': 0.0,
    'absorbRadius': 250e-6,
    'C': C_Glass,
    'rho': 2510.0
}

# Titanium Oxide
# HOC 852
C_TiO2 = array([[269.0, 177.0, 146.0, 0.0, 0.0, 0.0],
                [177.0, 269.0, 146.0, 0.0, 0.0, 0.0],
                [146.0, 146.0, 480.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 124.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 124.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 192.0]])

D_TiO2 = {
    'name': r'$Ti O_2$',
    'xy_length': 600e-6,  # 0.5*1500.0e-6,
    'z_length': 5e-3,
    'grid_pts': 2 ** 7,
    'z_grid_pts': 101,
    'f_0': 0.0,  # phonon frequency
    'kappa': 0.0,
    'absorbRadius': 300e-6,
    'C': C_TiO2,
    'rho': 4260.0
}

# Aluminum Nitride
# TODO: check tensor from COMSOL
C_AlN1 = array([[411.0, 149.0, 99.0, 0.0, 0.0, 0.0],
                [149.0, 410.0, 99.0, 0.0, 0.0, 0.0],
                [99.0, 99.0, 389.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 125.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 125.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 130.5]])

# From HOC p.611
C_AlN2 = array([[296.0, 130.0, 158.0, 0.0, 0.0, 0.0],
                [130.0, 296.0, 158.0, 0.0, 0.0, 0.0],
                [158.0, 158.0, 267.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 241.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 241.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 83.0]])

D_AlN = {
    'name': r'$Al N$',
    'xy_length': 600e-6,  # 0.5*1500.0e-6,
    'z_length': 5e-3,
    'grid_pts': 2 ** 7,
    'z_grid_pts': 101,
    'f_0': 0.0,  # phonon frequency
    'kappa': 0.0,
    'absorbRadius': 300e-6,
    'C': C_AlN2,
    'rho': 3255.0
}

# From HOC p.828
C_CaF2 = array([[165.0, 47.0, 47.0, 0.0, 0.0, 0.0],
                [47.0, 165.0, 47.0, 0.0, 0.0, 0.0],
                [47.0, 47.0, 165.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 33.9, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 33.9, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 33.9]])

D_CaF2 = {
    'name': r'$Ca F_2$',
    'xy_length': 600e-6,  # 0.5*1500.0e-6,
    'z_length': 5e-3,
    'grid_pts': 2 ** 7,
    'z_grid_pts': 101,
    'f_0': 13.303e9,  # Brillouin fr
    'kappa': 0.0,
    'absorbRadius': 250e-6,
    'C': C_CaF2,
    'rho': 3179.0
}

# Lithium niobate
# From HOC p. 848
C_LiNb03 = array([[202.0, 55.0, 72.0, 8.5, 0.0, 0.0],
                  [55.0, 202.0, 72.0, -8.5, 0.0, 0.0],
                  [72.0, 72.0, 244.0, 0.0, 0.0, 0.0],
                  [8.5, -8.5, 0.0, 60.2, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 60.2, 8.5],
                  [0.0, 0.0, 0.0, 0.0, 8.5, 0.5 * (202.0 - 55.0)]])
D_LiNb03 = {
    'name': r'$Li Nb O_3$',
    'xy_length': 600e-6,  # 0.5*1500.0e-6,
    'z_length': 5e-3,
    'grid_pts': 2 ** 7,
    'z_grid_pts': 101,
    'kappa': 0.0,
    'absorbRadius': 250e-6,
    'C': C_LiNb03,
    'rho': 4628.0
}

# Vacuum constants for electromagnetic beams

D_opt = {
    'name': 'Vacuum',
    'vl': 3e8,  # long velocity of bulk material
    'vt': 3e8,  # trans velocity of bulk material
    'vlPiezo': 3e8,  # previously called vlAlN, because AlN is the standard material we consider (with vlAlN=11050)
    'xy_length': 0.5 * 1500.0e-6,  # transversal xy_length (diameter) of simulation channel
    'z_length': 11.86e-3,  # length of crystal
    'grid_pts': 2 ** 7,
    # transversal gridpoints. This determines the resolution in r-space, and the max wavevector in k-space.
    'z_grid_pts': 101,  # longitudinal gridpoints
    'f_0': 193.4e12,  # frequency around which the sweep is executed (more parameters in D_sim_params)
    'kappa': 0,  # fraction of energy (or field?) lost at each 1/2 roundtrip
    'absorbRadius': 250e-6,
    # radius beyond which the beam gets damped on the transversal edge of the simulation channel
    'expCorrFactor': 1  # 26.6, } # from Mathematica
}


def getVelocities(stiffness, density, q):
    """
    Returns the phase velocities of the christoffel equation
    """
    christoffel = ch.Christoffel(stiffness, density)
    christoffel.set_direction_cartesian(q)
    return christoffel.get_phase_velocity()


def getquasiLongPolarization(stiffness, density, q):
    """
    Returns the polarization of the (quasi)-longitudinal wave
    """
    christoffel = ch.Christoffel(stiffness, density)
    christoffel.set_direction_cartesian(q)
    return christoffel.get_eigenvec()[2]


def getLongitudinalVelocity(stiffness, density):
    """
    Return vl for q=[0,0,1]
    """
    return max(getVelocities(stiffness, density, array([0, 0, 1])))


def getTransverseVelocity(stiffness, density):
    """
    Returns vt in case of symmetry around qz axis? for q=[0,0,1]
    """
    return min(getVelocities(stiffness, density, array([0, 0, 1])))


def surface_plotter(qx: ndarray,
                    qy: ndarray,
                    qz: ndarray,
                    interpol_func: Callable,
                    material_name: str = 'Unknown material'):

    # Axes projections
    fig, axs = subplots(ncols=2, nrows=1, facecolor='white', figsize=(18.0, 8.0))

    x = linspace(-2e-4, 2e-4, 1000)

    # ky - kz projections
    y_sel = isclose(qx, zeros_like(qx), atol=1e-7)
    z_sel = isclose(qx, zeros_like(qx), atol=1e-7)
    axs[0].plot(extract(y_sel[0, :], qy[0, :]), extract(z_sel[0, :], qz[0, :]), marker='.', linestyle=' ', color='C0',
                label='Quasi-Transverse')
    axs[0].plot(extract(y_sel[1, :], qy[1, :]), extract(z_sel[1, :], qz[1, :]), marker='.', linestyle=' ', color='C0')
    axs[0].plot(extract(y_sel[2, :], qy[2, :]), extract(z_sel[2, :], qz[2, :]), marker='.', linestyle=' ', color='C1',
                label='Quasi-Longitudinal')
    axs[0].plot(x, interpol_func(zeros_like(x), x), linestyle='--', color='C2', label='Interpolation')
    axs[0].set_xlabel(r'$q_{y} \ / \ \omega$')
    axs[0].set_ylabel(r'$q_{z} \ / \ \omega$')
    axs[0].legend(loc='best')

    # kx - kz projections
    x_sel = isclose(qy, zeros_like(qy), atol=1e-7)
    z_sel = isclose(qy, zeros_like(qy), atol=1e-7)
    axs[1].plot(extract(x_sel[0, :], qx[0, :]), extract(z_sel[0, :], qz[0, :]), marker='.', linestyle=' ', color='C0',
                label='Quasi-Transverse')
    axs[1].plot(extract(x_sel[1, :], qx[1, :]), extract(z_sel[1, :], qz[1, :]), marker='.', linestyle=' ', color='C0')
    axs[1].plot(extract(x_sel[2, :], qx[2, :]), extract(z_sel[2, :], qz[2, :]), marker='.', linestyle=' ', color='C1',
                label='Quasi-Longitudinal')
    axs[1].plot(x, interpol_func(x, zeros_like(x)), linestyle='--', color='C2', label='Interpolation')
    axs[1].set_xlabel(r'$q_{x} \ / \ \omega$')
    axs[1].set_ylabel(r'$q_{z} \ / \ \omega$')
    axs[1].legend(loc='best')

    fig.suptitle(fr"Slowness surface diagnostics in {material_name}")
    fig.tight_layout()

    return fig, axs


def surface_2d_interpolator(material: dict, max_azimuth_angle: float = 1.1, interpol_method='cubic', plotting=False) ->\
        Tuple[Union[NearestNDInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator], float, Optional[Figure]]:
    """

    @param material: Dictionary containing material properties
    @param max_azimuth_angle: Maximum azimuthal angle for the sample
    @param interpol_method: Interploation method ('nearest', 'linear' and 'cubic')
    @param plotting: Whether you wish to make the surface equation plotters
    @return: Interpolator and Figure containing slowness surfaces projected at the axes
    """

    n_theta = 15  # azimuthal angle
    n_phi = 21  # polar angle
    n_total = n_theta * n_phi

    theta = linspace(0, max_azimuth_angle, n_theta)  # azimuthal angle
    phi = linspace(0, 2 * pi, n_phi)  # polar angle

    radius = zeros(shape=3)
    qx = zeros(shape=(3, n_total))
    qy = zeros(shape=(3, n_total))
    qz = zeros(shape=(3, n_total))

    christoffel = ch.Christoffel(stiffness=material['C'], density=material['rho'])
    christoffel.set_direction_spherical(0.0, 0.0)
    v_long_z = 1e3 * christoffel.get_phase_velocity()[2]

    with nditer(op=[qx, qy, qz], flags=['external_loop'], op_flags=['readwrite'], order='F') as it:
        for (t, p), (x, y, z) in zip(itertools_product(theta, phi), it):

            christoffel.set_direction_spherical(t, p)
            radius[...] = 1e-3 / christoffel.get_phase_velocity()
            x[...] = radius * sin(t) * cos(p)
            y[...] = radius * sin(t) * sin(p)
            z[...] = radius * cos(t)

    interpol_points = list(zip(qx[2, (n_theta - 1):], qy[2, (n_theta - 1):]))
    if interpol_method == 'nearest':
        func = NearestNDInterpolator(x=interpol_points, y=qz[2, (n_theta - 1):])
    elif interpol_method == 'linear':
        func = LinearNDInterpolator(points=interpol_points, values=qz[2, (n_theta - 1):])
    elif interpol_method == 'cubic':
        func = CloughTocher2DInterpolator(points=interpol_points, values=qz[2, (n_theta - 1):])
    else:
        raise ValueError(f"Invalid interpolation method: {interpol_method}\n"
                         f"Valid args are 'nearest', 'linear' and 'cubic'")

    if plotting:
        fig_projections = surface_plotter(qx=qx, qy=qy, qz=qz, interpol_func=func, material_name=material['name'])
    else:
        fig_projections = None

    return func, v_long_z, fig_projections


def surface_paraxial_params(material: dict, theta_interpol: float = 0.5, num_samples: int = 55, plotting=False):

    christoffel = ch.Christoffel(stiffness=material['C'], density=material['rho'])

    theta_arr = linspace(start=-theta_interpol, stop=theta_interpol, num=num_samples, endpoint=True)
    phi_arr = array([0.0, 0.5*pi, 0.25*pi])

    qxy = zeros(shape=(3, num_samples))
    qz = zeros(shape=(3, num_samples))

    with nditer(op=[qxy, qz], op_flags=['readwrite'], order='C') as it:
        for (id_p, p), t in itertools_product(enumerate(phi_arr), theta_arr):

            christoffel.set_direction_spherical(theta=t, phi=p)
            r = 1e-3 / christoffel.get_phase_velocity()[2]
            it[1][...] = r * cos(t)

            if id_p % 2 == 0:
                it[0][...] = r * sin(t) * cos(p)
            else:
                it[0][...] = r * sin(t) * sin(p)

            it.iternext()

    spline_x = splrep(x=qxy[0, :], y=qz[0, :])
    spline_y = splrep(x=qxy[1, :], y=qz[1, :])
    spline_xy = splrep(x=qxy[2, :], y=qz[2, :])

    christoffel.set_direction_spherical(theta=0.0, phi=0.0)
    s0 = 1e-3 / christoffel.get_phase_velocity()[2]
    a = splev(x=0.0, tck=spline_x, der=1)
    b = splev(x=0.0, tck=spline_y, der=1)
    dx = 0.5 * splev(x=0.0, tck=spline_x, der=2)
    dy = 0.5 * splev(x=0.0, tck=spline_y, der=2)
    c = 0.5 * splev(x=0.0, tck=spline_xy, der=2) - dx - dy

    if plotting:
        x_fine = linspace(start=min(qxy[0, :]), stop=max(qxy[0, ]), num=1000, endpoint=True)
        fig, axs = subplots(nrows=1, ncols=3, facecolor='white', figsize=(18.0, 8.0))

        axs[0].plot(qxy[0, :], qz[0, :], label='Sampled points', color='C1', marker='.', linestyle=' ')
        axs[0].plot(x_fine, splev(x_fine, spline_x), label='Interpolation', color='C2', linestyle='--')

        axs[1].plot(qxy[1, :], qz[1, :], label='Sampled points', color='C1', marker='.', linestyle=' ')
        axs[1].plot(x_fine, splev(x_fine, spline_y), label='Interpolation', color='C2', linestyle='--')

        axs[2].plot(qxy[2, :], qz[2, :], label='Sampled points', color='C1', marker='.', linestyle=' ')
        axs[2].plot(x_fine, splev(x_fine, spline_xy), label='Interpolation', color='C2', linestyle='--')

        axs[0].legend(loc='best')
        axs[0].set_xlabel(r'$q_{x} \ / \ \omega$')
        axs[0].set_ylabel(r'$q_{z} \ / \ \omega$')

        axs[1].legend(loc='best')
        axs[1].set_xlabel(r'$q_{y} \ / \ \omega$')
        axs[1].set_ylabel(r'$q_{z} \ / \ \omega$')

        axs[2].legend(loc='best')
        axs[2].set_xlabel(r'$q_{xy} \ / \ \omega$')
        axs[2].set_ylabel(r'$q_{z} \ / \ \omega$')

        fig.suptitle(f'{material["name"]}')
        fig.tight_layout()

    else:
        fig = None

    return s0, float(a), float(b), c, dx, dy, fig


def effective_transverse_velocities(material: dict):
    s0, _, _, _, dx, dy, _ = surface_paraxial_params(material=material)
    v_eff_x = (-2.0 * dx / s0) ** 0.5
    v_eff_y = (-2.0 * dy / s0) ** 0.5

    return v_eff_x, v_eff_y


if __name__ == '__main__':
    D_list = [
        D_Sapphire,
        D_Quartz,
        D_GaAs,
        D_Glass,
        D_TiO2,
        D_AlN,
        D_CaF2,
        D_LiNb03
    ]

    for d in D_list:
        S0, A, B, C, DX, DY, _ = surface_paraxial_params(material=d)
        vx, vy = effective_transverse_velocities(material=d)
        v_long_z = 1.0 / S0

        christoffel = ch.Christoffel(stiffness=d['C'], density=d['rho'])
        christoffel.set_direction_spherical(0.5 * pi, 0.0)
        vx_true = 1e3 * christoffel.get_phase_velocity()[2]

        print(f'Material: {d["name"]}\n')
        print(f'S0: {S0:.2e}\t A: {A:.2e}\t B: {B:.2e}\t C: {C:.2e} \t Dx: {DX:.2e}    Dy: {DY:.2e}')
        print(f'vx_true: {vx_true:.3e} \t vx_eff: {vx:.3e} \t v_z {v_long_z:.3e} \n')
