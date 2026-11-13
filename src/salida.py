import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTADOS = Path(__file__).resolve().parent.parent / "results"
RESULTADOS.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 9})


def guardar_figura(fig, nombre):
    fig.savefig(RESULTADOS / nombre)
    plt.close(fig)


def cientifica(x):
    mantisa, exponente = f"{x:.2e}".split("e")
    return rf"${mantisa}\cdot10^{{{int(exponente)}}}$"


def guardar_tabla(nombre, encabezados, filas, encabezados_tex=None, filas_tex=None):
    with open(RESULTADOS / f"{nombre}.csv", "w", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(encabezados)
        escritor.writerows(filas)

    encabezados_tex = encabezados_tex or encabezados
    filas_tex = filas_tex or filas
    lineas = [r"\begin{tabular}{" + "l" + "r" * (len(encabezados_tex) - 1) + "}", r"\toprule",
              " & ".join(encabezados_tex) + r" \\", r"\midrule"]
    lineas += [" & ".join(str(celda) for celda in fila) + r" \\" for fila in filas_tex]
    lineas += [r"\bottomrule", r"\end{tabular}"]
    (RESULTADOS / f"{nombre}.tex").write_text("\n".join(lineas) + "\n")
