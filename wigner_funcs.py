from numpy import linspace
from matplotlib.pyplot import subplots
from qutip import coherent_dm, fock_dm, qobj
from qutip.wigner import wigner, qfunc


def generate_wigner_plots(rho_fock: qobj,
                          rho_coherent: qobj,
                          rho_cat: qobj) -> None:

    fig, axs = subplots(nrows=1, ncols=3, figsize=(9, 3))
    xvec = linspace(-6, 6, 300)
    zlevels = linspace(-0.4, 0.4, 50, endpoint=True)
    cmap = "seismic"

    zmap_fock = wigner(rho_fock, xvec=xvec, yvec=xvec)
    axs[0].contourf(xvec, xvec, zmap_fock, zlevels, cmap=cmap)
    axs[0].set_title("Fock state")
    del zmap_fock

    zmap_coherent = wigner(rho_coherent, xvec=xvec, yvec=xvec)
    axs[1].contourf(xvec, xvec, zmap_coherent, zlevels, cmap=cmap)
    axs[1].set_title("Coherent state")
    del zmap_coherent

    zmap_cat = wigner(rho_cat, xvec=xvec, yvec=xvec)
    pcm = axs[2].contourf(xvec, xvec, zmap_cat, zlevels, cmap=cmap)
    axs[2].set_title("Even cat state")
    del zmap_cat

    # fig.colorbar(pcm, ax=axs[:], location="right", shrink=0.9)
    fig.savefig(fname=f"./img/02_WignerFuncs.pdf")


def generate_husimi_plots(rho_fock: qobj,
                          rho_coherent: qobj,
                          rho_cat: qobj) -> None:

    fig, axs = subplots(nrows=1, ncols=3, figsize=(9, 3))
    xvec = linspace(-6, 6, 300)
    zlevels = linspace(-0.5, 0.5, 50)
    cmap = "seismic"

    zmap_fock = qfunc(rho_fock, xvec=xvec, yvec=xvec)
    axs[0].contourf(xvec, xvec, zmap_fock, zlevels, cmap=cmap)
    axs[0].set_title("Fock state")
    del zmap_fock

    zmap_coherent = qfunc(rho_coherent, xvec=xvec, yvec=xvec)
    pcm = axs[1].contourf(xvec, xvec, zmap_coherent, zlevels, cmap=cmap)
    axs[1].set_title("Coherent state")
    del zmap_coherent

    zmap_cat = qfunc(rho_cat, xvec=xvec, yvec=xvec)
    pcm = axs[2].contourf(xvec, xvec, zmap_cat, zlevels, cmap=cmap)
    axs[2].set_title("Even cat state")
    del zmap_cat

    fig.colorbar(pcm, ax=axs[:], location="right", shrink=0.9)
    fig.savefig(fname=f"./img/02_HusimiQfuncs.pdf")


if __name__ == "__main__":

    N = 20
    alpha = 3 ** 0.5

    rho_coherent = coherent_dm(N, alpha)
    rho_fock = fock_dm(N, 3)
    rho_even_cat = coherent_dm(N, alpha) + coherent_dm(N, -alpha)

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
