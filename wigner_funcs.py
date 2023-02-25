from numpy import linspace, amax, amin
from matplotlib.pyplot import figure
from mpl_toolkits.axes_grid1 import ImageGrid
from qutip import fock_dm, qobj, basis, displace
from qutip.wigner import wigner, qfunc


def generate_wigner_plots(rho_fock: qobj,
                          rho_coherent: qobj,
                          rho_cat: qobj) -> None:

    fig = figure(figsize=(8,3), facecolor="white")
    grid = ImageGrid(
        fig=fig,
        rect=111,
        nrows_ncols=(1,3),
        axes_pad=0.3,
        share_all=False,
        cbar_mode="single",
        cbar_location="right",
        cbar_pad=0.5,
        cbar_size="5%"
    )

    xvec = linspace(-8, 8, 300)
    zlevels = linspace(-0.5, 0.5, 100, endpoint=False)
    zmap_fock = wigner(rho_fock, xvec=xvec, yvec=xvec)
    zmap_coherent = wigner(rho_coherent, xvec=xvec, yvec=xvec)
    zmap_cat = wigner(rho_cat, xvec=xvec, yvec=xvec)


    for ax, zm in zip(grid, [zmap_fock, zmap_coherent, zmap_cat]):
        m = max(-amin(zm), amax(zm))
        
        plt1 = ax.contourf(xvec, xvec, zm, zlevels, cmap="seismic")
        ax.set_xlabel("I")
        ax.set_aspect(1)

    grid.cbar_axes[0].colorbar(plt1)
    grid[0].set_ylabel("Q")
    grid[0].set_title("Fock state $\\left|3\\right>$")
    grid[1].set_title("Coherent state\n$\\alpha = 2 + 2i$")
    grid[2].set_title("Even cat state\n$\\alpha = 2 + 2i$")
    
    fig.tight_layout()
    fig.savefig(fname=f"./img/WignerFuncs.pdf")


def generate_husimi_plots(rho_fock: qobj,
                          rho_coherent: qobj,
                          rho_cat: qobj) -> None:


    fig = figure(figsize=(8,3), facecolor="white")

    grid = ImageGrid(
        fig=fig,
        rect=111,
        nrows_ncols=(1,3),
        axes_pad=0.3,
        share_all=False,
        cbar_mode="single",
        cbar_location="right",
        cbar_pad=0.5,
        cbar_size="5%"
    )

    xvec = linspace(-8, 8, 300)
    zlevels = linspace(-0.2, 0.2, 100, endpoint=False)
    zmap_fock = qfunc(rho_fock, xvec=xvec, yvec=xvec)
    zmap_coherent = qfunc(rho_coherent, xvec=xvec, yvec=xvec)
    zmap_cat = qfunc(rho_cat, xvec=xvec, yvec=xvec)

    for ax, zm in zip(grid, [zmap_fock, zmap_coherent, zmap_cat]):
        m = max(-amin(zm), amax(zm))
        
        plt1 = ax.contourf(xvec, xvec, zm, zlevels, cmap="seismic")
        ax.set_xlabel("I")
        ax.set_aspect(1)

    grid.cbar_axes[0].colorbar(plt1)
    grid[0].set_ylabel("Q")
    grid[0].set_title("Fock state $\\left|3\\right>$")
    grid[1].set_title("Coherent state\n$\\alpha = 2 + 2i$")
    grid[2].set_title("Even cat state\n$\\alpha = 2 + 2i$")
    
    fig.tight_layout()
    fig.savefig(fname=f"./img/HusimiQfuncs.pdf")


def generate_wigner_husimi():

    N = 30
    alpha = 2 + 2j
    D1 = displace(N=N, alpha=alpha)
    D2 = displace(N=N, alpha=-alpha)

    rho_fock = fock_dm(N, 3)
    rho_coherent = D1 * basis(N, 0)
    rho_even_cat = (D1 + D2) * basis(N, 0)

    generate_wigner_plots(
        rho_fock=rho_fock,
        rho_coherent=rho_coherent,
        rho_cat=rho_even_cat
    )

    generate_husimi_plots(
        rho_fock=rho_fock,
        rho_coherent=rho_coherent,
        rho_cat=rho_even_cat
    )

if __name__ == "__main__":
    generate_wigner_husimi()
