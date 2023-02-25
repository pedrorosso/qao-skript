from schemdraw import Drawing
from schemdraw.elements import Annotate, Capacitor, EncircleBox, Inductor, Line
from schemdraw.elements.twoterm import Josephson


def draw_capacitive_coupler() -> None:

    with Drawing() as d:

        d.config(unit=3)
        d += Line().left(3).dot()
        d += (Lr := Inductor().down().label(r"$L_r$"))
        d += Line().right(3)
        d += (Cr := Capacitor().up().label(r"$C_r$").dot())
        d += (Cc := Capacitor().right().label(r"$C_c$"))
        d += (JJ := Josephson().down())
        d += Line().right(3)
        d += (Cq := Capacitor().up().label(r"$C_q$"))
        d += Line().left(3).dot()

        d += (Vr := Line().left(1).at(Lr.start).dot(open=True).label(r"$V_r$", "left"))
        d += (Vq := Line().right(1).at(Cq.end).dot(open=True).label(r"$V_q$", "right"))

        d += (resonator := EncircleBox([Lr, Cr, Vr],
              padx=0.2).linestyle("--").linewidth(1).color("green"))
        d += Annotate().at(resonator.S).delta(dx=0,
                                              dy=-.5).label("LC Resonator").color("green")
        d += (qubit := EncircleBox([JJ, Cq, Vq],
              padx=0.2).linestyle("--").linewidth(1).color("blue"))
        d += Annotate().at(qubit.S).delta(dx=0, dy=-.5).label("Qubit").color("blue")
        d += (coupler := EncircleBox([Cc], padx=-0.6)
              ).linestyle("--").linewidth(1).color("red")
        d += Annotate().at(coupler.N).delta(dx=0, dy=.5).label("Coupling").color("red")

        d.save(fname="img/SC_Capacitive.png", dpi=600)


if __name__ == "__main__":
    draw_capacitive_coupler()
