from schemdraw import Drawing
from schemdraw.elements import Annotate, Capacitor, EncircleBox, Inductor, Line, Label
from schemdraw.elements.twoterm import Josephson
from schemdraw.elements.lines import LoopCurrent


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


def draw_squid_diagram() -> None:

    with Drawing() as d:

        d.config(unit=3)
        d += (LeftLine := Line().up(3))
        d += (TopJJ := Josephson().right().label(r"$I_1 \ \rightarrow$", "top"))
        d += (RightLine := Line().down(3))
        d += (BottomJJ := Josephson().left().label(r"$I_2 \ \rightarrow$", "bottom"))

        d += Line().at(LeftLine.center).left(1.5).label(r"$I \ \rightarrow$", "top")
        d += Line().right(1.5).at(RightLine.center)
        d += LoopCurrent(elm_list=[TopJJ, RightLine, BottomJJ, LeftLine],
                         direction="ccw",
                         theta1=150,
                         theta2=210).label(label=r"$I_S$", ofst=(-0.8, 0.0))
        RightLine.add_label(label="\u2299 B", color="Blue", loc="top", size=20)

        d.save(fname="img/SQUID.png", dpi=600)


def freq_tunable_transmon():

    with Drawing() as d:

        d.config(unit=2)
        d += (TopLine := Line().right(2))
        d += Josephson().down()
        d += Line().left(2)
        d += Josephson().up()

        d += Line().up(0.5).at(TopLine.center)
        d += Line().right(4)
        d += Capacitor().down(3)
        d += Line().left(4)
        d += Line().up(0.5)

        d.save(fname="img/FreqTunableTransmon.png", dpi=600)


def draw_all_circuits():
    draw_capacitive_coupler()
    draw_squid_diagram()
    freq_tunable_transmon()


if __name__ == "__main__":
    draw_all_circuits()
