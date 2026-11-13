import json

import matplotlib.pyplot as plt
import numpy as np

from modelo import NOMINAL, T_FINAL, euler
from salida import RESULTADOS, cientifica, guardar_figura, guardar_tabla

PUNTOS_POR_MES = 128
K_REF = 14
PASOS_DUPLICADOS = [2.0**k for k in range(-7, 3)]
PASOS_INTERMEDIOS = [0.75, 1.25, 1.5]


def referencia(k):
    n_pasos = int(T_FINAL) * 2**k
    return euler(NOMINAL, n_pasos, guardar_cada=2**k // PUNTOS_POR_MES)


def errores_maximos(h, t_ref, y_ref):
    t, y = euler(NOMINAL, round(T_FINAL / h))
    salto = round(h * PUNTOS_POR_MES)
    assert np.allclose(t, t_ref[::salto])
    if not np.isfinite(y).all():
        return np.full(3, np.inf), t, y
    return np.abs(y - y_ref[::salto]).max(axis=0), t, y


def main():
    resumen = {}

    t_ref, y_ref = referencia(K_REF)
    _, y_fina = referencia(K_REF + 1)
    diferencia = np.abs(y_ref - y_fina).max(axis=0)
    resumen["h_ref"] = 2.0**-K_REF
    resumen["dif_ref_vs_fina_T"] = diferencia[0]
    resumen["error_estimado_ref_T"] = 2 * diferencia[0]

    resultados = {}
    for h in sorted(PASOS_DUPLICADOS + PASOS_INTERMEDIOS):
        resultados[h] = errores_maximos(h, t_ref, y_ref)

    filas, filas_tex = [], []
    for h in PASOS_DUPLICADOS:
        err = resultados[h][0]
        anterior = resultados.get(h / 2)
        if np.isfinite(err[0]) and anterior is not None:
            cociente = err[0] / anterior[0][0]
            orden = np.log(cociente) / np.log(2)
            cociente, orden = f"{cociente:.3f}", f"{orden:.3f}"
        else:
            cociente = orden = "--"
        if np.isfinite(err[0]):
            filas.append([h, round(T_FINAL / h), *(f"{e:.4e}" for e in err), cociente, orden])
            filas_tex.append([f"{h:g}", round(T_FINAL / h), *(cientifica(e) for e in err), cociente, orden])
        else:
            filas.append([h, round(T_FINAL / h), "inf", "inf", "inf", cociente, orden])
            filas_tex.append([f"{h:g}", round(T_FINAL / h), *["diverge"] * 3, cociente, orden])
    guardar_tabla(
        "tabla_convergencia",
        ["h", "N", "Err_T", "Err_H", "Err_P", "Err_T(h)/Err_T(h/2)", "p"], filas,
        ["$h$ [mes]", "$N$", r"$\mathrm{Err}_T$ [°C]", r"$\mathrm{Err}_H$ [u.h.]",
         r"$\mathrm{Err}_P$ [u.p.]", r"$\frac{\mathrm{Err}_T(h)}{\mathrm{Err}_T(h/2)}$", "$p$"],
        filas_tex)
    resumen["error_T_menor_paso"] = resultados[PASOS_DUPLICADOS[0]][0][0]
    resumen["relacion_ref_menor_paso"] = resumen["error_estimado_ref_T"] / resumen["error_T_menor_paso"]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    hs = np.array([h for h in sorted(resultados) if np.isfinite(resultados[h][0][0])])
    for i, (nombre, marca) in enumerate([("T [°C]", "o-"), ("H [u.h.]", "s--"), ("P [u.p.]", "^--")]):
        ax.loglog(hs, [resultados[h][0][i] for h in hs], marca, ms=4, lw=1, label=f"Err de {nombre}")
    ax.loglog(hs, resultados[hs[0]][0][0] * hs / hs[0], "k:", label="pendiente 1")
    divergentes = [h for h in sorted(resultados) if not np.isfinite(resultados[h][0][0])]
    for h in divergentes:
        ax.axvline(h, color="r", lw=0.8, ls="-.")
    ax.axvline(divergentes[0], color="r", lw=0.8, ls="-.", label="diverge (overflow)")
    ax.set_xlabel("h [mes]")
    ax.set_ylabel("máximo error absoluto")
    ax.set_title("Error de Euler respecto de la referencia")
    ax.legend(fontsize=7)
    guardar_figura(fig, "fig_convergencia.png")

    ultimo_ciclo_ref = t_ref >= 48
    filas, filas_tex = [], []
    for h in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0]:
        err, t, y = resultados[h]
        if np.isfinite(err[0]):
            P_ciclo = y[t >= 48, 2]
            fila = [f"{h:g}", f"{err[0]:.3f}", f"{err[2]:.2f}", f"{P_ciclo.min():.2f}", f"{P_ciclo.max():.2f}"]
        else:
            with np.errstate(invalid="ignore"):
                t_explota = t[np.argmax(~(np.abs(y[:, 0]) < 100))]
            fila = [f"{h:g}", "diverge", "diverge", "--", f"$|T|>100$ en $t={t_explota:g}$"]
        filas.append(fila)
    P_ref = y_ref[ultimo_ciclo_ref, 2]
    filas.insert(0, ["ref.", "0", "0", f"{P_ref.min():.2f}", f"{P_ref.max():.2f}"])
    guardar_tabla(
        "tabla_estabilidad", ["h", "Err_T", "Err_P", "P_min ultimo ciclo", "P_max ultimo ciclo"], filas,
        ["$h$ [mes]", r"$\mathrm{Err}_T$ [°C]", r"$\mathrm{Err}_P$ [u.p.]",
         r"$P_{\min}$ en $[48,60]$", r"$P_{\max}$ en $[48,60]$"])

    fig, ejes = plt.subplots(2, 2, figsize=(8, 5), sharex=True)
    for ax, h in zip(ejes.flat, [0.5, 1.0, 1.25, 1.5]):
        _, t, y = resultados[h]
        ax.plot(t_ref, y_ref[:, 2], "k-", lw=1, label="referencia")
        ax.plot(t, y[:, 2], "r.-", lw=0.7, ms=3, label=f"Euler h = {h:g}")
        ax.set_ylim(-2, 25)
        ax.set_title(f"h = {h:g} mes")
        ax.legend(fontsize=7, loc="upper left")
    for ax in ejes[1]:
        ax.set_xlabel("t [mes]")
    for ax in ejes[:, 0]:
        ax.set_ylabel("P [u.p.]")
    fig.suptitle("P(t) con pasos cercanos al límite de estabilidad")
    guardar_figura(fig, "fig_inestabilidad.png")

    (RESULTADOS / "parte1_resumen.json").write_text(json.dumps(resumen, indent=2))
    print(json.dumps(resumen, indent=2))


if __name__ == "__main__":
    main()
