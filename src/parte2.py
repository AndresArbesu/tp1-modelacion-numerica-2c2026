import json

import matplotlib.pyplot as plt
import numpy as np

from indicadores import T_CICLO, error_periodicidad, indicadores, integral_trapecios
from modelo import NOMINAL, con_parametros, euler
from salida import RESULTADOS, guardar_figura, guardar_tabla

N_PASOS = 7680

RANGOS = {"A": (0.25, 0.60), "beta": (0.10, 0.30), "gamma": (0.05, 0.20), "sigma": (0.40, 1.00)}
SIMBOLO = {"A": "A", "beta": r"\beta", "gamma": r"\gamma", "sigma": r"\sigma"}
UNIDAD = {"A": "mes$^{-1}$", "beta": "u.h. mes$^{-1}$ °C$^{-2}$",
          "gamma": "mes$^{-1}$ u.p.$^{-1}$", "sigma": "mes$^{-1}$"}

BARRIDO = {"A": 15, "beta": 21, "gamma": 16, "sigma": 13}
GRILLA = {"A": 8, "beta": 9, "gamma": 7, "sigma": 7}

COTA = 100.0
TOL_PERIODICIDAD = 0.01


def valores(nombre, cantidad):
    return np.round(np.linspace(*RANGOS[nombre], cantidad), 6)


def evaluar(params, n_pasos=N_PASOS):
    t, y = euler(params, n_pasos)
    with np.errstate(invalid="ignore"):
        res = indicadores(t, y)
        res["periodicidad"] = error_periodicidad(t, y)
        res["maximo"] = np.abs(y).max(axis=(0, 1))
    ciclo = t >= T_CICLO
    res["Tmin"] = y[ciclo, 0].min(axis=0)
    res["Tmax"] = y[ciclo, 0].max(axis=0)
    res["Hmed"] = integral_trapecios(t[ciclo], y[ciclo, 1]) / (t[-1] - T_CICLO)
    return res


def es_valido(res):
    with np.errstate(invalid="ignore"):
        return (res["maximo"] < COTA) & (res["periodicidad"] < TOL_PERIODICIDAD)


def buscar_en_grilla(ejes):
    filas = []
    B, G, S = (m.ravel() for m in np.meshgrid(ejes["beta"], ejes["gamma"], ejes["sigma"], indexing="ij"))
    for A in ejes["A"]:
        params = con_parametros(A=A, beta=B, gamma=G, sigma=S)
        res = evaluar(params)
        valido = es_valido(res)
        filas += zip(np.full(B.size, A), B, G, S, res["SP"], res["desfase"],
                     res["periodicidad"], res["maximo"], res["t_Pmax"] > res["t_Tmax"], valido)
    return np.array(filas)


def escenario_nominal(resumen):
    t, y = euler(NOMINAL, N_PASOS)
    res = evaluar(NOMINAL)
    resumen["nominal"] = {k: float(v) for k, v in res.items()}
    ciclo = t >= T_CICLO
    resumen["nominal"]["t_Hmax"] = float(t[ciclo][y[ciclo, 1].argmax()])

    filas = []
    for n in [N_PASOS // 8, N_PASOS // 4, N_PASOS // 2, N_PASOS, 2 * N_PASOS, 4 * N_PASOS, 8 * N_PASOS]:
        r = evaluar(NOMINAL, n)
        filas.append([f"1/{n // 60}", f"{r['SP']:.4f}", f"{r['desfase']:.4f}", f"{r['Pmax']:.4f}",
                      f"{r['Pmin']:.4f}", f"{r['Pmed']:.4f}"])
    guardar_tabla("tabla_paso_indicadores", ["h", "SP", "desfase", "Pmax", "Pmin", "Pmed"], filas,
                  ["$h$ [mes]", "$S_P$", r"$\Delta t_{TP}$ [mes]", r"$P_{\max}$", r"$P_{\min}$", r"$P_{med}$"])

    nombres = ["T [°C]", "H [u.h.]", "P [u.p.]"]
    fig, ejes = plt.subplots(3, 1, figsize=(7, 5.5), sharex=True)
    for i, ax in enumerate(ejes):
        ax.plot(t, y[:, i], lw=1)
        ax.axvspan(48, 60, color="gray", alpha=0.15)
        ax.set_ylabel(nombres[i])
    ejes[0].set_title("Escenario nominal (sombreado: último ciclo anual)")
    ejes[2].set_xlabel("t [mes]")
    ejes[2].set_xticks(range(0, 61, 6))
    guardar_figura(fig, "fig_nominal_60.png")

    fig, ejes = plt.subplots(3, 1, figsize=(7, 5.5), sharex=True)
    for i, ax in enumerate(ejes):
        ax.plot(t[ciclo], y[ciclo, i], lw=1)
        ax.set_ylabel(nombres[i])
    ejes[0].axvline(res["t_Tmax"], color="r", ls="--", lw=0.8, label=f"máx. de T: t = {res['t_Tmax']:.2f}")
    ejes[2].axvline(res["t_Tmax"], color="r", ls="--", lw=0.8)
    ejes[2].axvline(res["t_Pmax"], color="g", ls="--", lw=0.8, label=f"máx. de P: t = {res['t_Pmax']:.2f}")
    ejes[2].axhline(res["Pmed"], color="k", ls=":", lw=0.8, label=f"Pmed = {res['Pmed']:.2f}")
    ejes[0].legend(fontsize=7)
    ejes[2].legend(fontsize=7, loc="lower left")
    ejes[0].set_title(f"Último ciclo anual: SP = {res['SP']:.3f}, desfase = {res['desfase']:.2f} meses")
    ejes[2].set_xlabel("t [mes]")
    ejes[2].set_xticks(range(48, 61))
    guardar_figura(fig, "fig_nominal_ciclo.png")


def sensibilidad(resumen):
    fig, ejes = plt.subplots(2, 4, figsize=(11, 4.8), sharex="col")
    filas_csv, filas_tex = [], []
    for columna, nombre in enumerate(RANGOS):
        q = valores(nombre, BARRIDO[nombre])
        res = evaluar(con_parametros(**{nombre: q}))
        assert es_valido(res).all()

        for i in range(q.size):
            filas_csv.append([nombre, q[i]] + [f"{res[k][i]:.5f}" for k in
                             ("SP", "desfase", "Pmax", "Pmin", "Pmed", "Tmin", "Tmax", "Hmed")])
        for i in (0, -1):
            filas_tex.append([f"${SIMBOLO[nombre]} = {q[i]:.2f}$"] + [f"{res[k][i]:.3f}" for k in ("SP", "desfase")]
                             + [f"{res[k][i]:.2f}" for k in ("Pmax", "Pmin", "Pmed", "Tmin", "Tmax", "Hmed")])
        resumen["sensibilidad"][nombre] = {
            "SP_monotona": bool(np.all(np.diff(res["SP"]) > 0) or np.all(np.diff(res["SP"]) < 0)),
            "desfase_monotono": bool(np.all(np.diff(res["desfase"]) > 0) or np.all(np.diff(res["desfase"]) < 0)),
            "SP_extremos": [float(res["SP"][0]), float(res["SP"][-1])],
            "desfase_extremos": [float(res["desfase"][0]), float(res["desfase"][-1])],
            "desfase_min": float(res["desfase"].min()),
            "q_desfase_min": float(q[res["desfase"].argmin()]),
        }

        ejes[0, columna].plot(q, res["SP"], "o-", ms=3, lw=1)
        ejes[1, columna].plot(q, res["desfase"], "o-", ms=3, lw=1, color="C1")
        for ax in ejes[:, columna]:
            ax.axvline(NOMINAL[nombre], color="gray", ls=":", lw=0.8)
        ejes[0, columna].set_title(f"${SIMBOLO[nombre]}$")
        ejes[1, columna].set_xlabel(f"${SIMBOLO[nombre]}$ [{UNIDAD[nombre]}]")
    ejes[0, 0].set_ylabel("$S_P$")
    ejes[1, 0].set_ylabel(r"$\Delta t_{TP}$ [mes]")
    for ax in ejes[0]:
        ax.set_ylim(0.15, 0.70)
    for ax in ejes[1]:
        ax.set_ylim(0.6, 1.4)
    fig.suptitle("Sensibilidad de $S_P$ y $\\Delta t_{TP}$ (línea punteada: valor nominal)")
    guardar_figura(fig, "fig_sensibilidad.png")

    encabezados = ["parametro", "valor", "SP", "desfase", "Pmax", "Pmin", "Pmed", "Tmin", "Tmax", "Hmed"]
    guardar_tabla("tabla_sensibilidad", encabezados, filas_csv,
                  ["Valor", "$S_P$", r"$\Delta t_{TP}$", r"$P_{\max}$", r"$P_{\min}$", r"$P_{med}$",
                   r"$T_{\min}$", r"$T_{\max}$", r"$H_{med}$"], filas_tex)


def maximizar(resumen):
    ejes = {nombre: valores(nombre, GRILLA[nombre]) for nombre in RANGOS}
    grilla = buscar_en_grilla(ejes)
    columnas = ["A", "beta", "gamma", "sigma", "SP", "desfase", "periodicidad", "maximo",
                "maxP_despues_de_maxT", "valido"]
    np.savetxt(RESULTADOS / "grilla.csv", grilla, delimiter=",", header=",".join(columnas), comments="", fmt="%.6g")

    validas = grilla[grilla[:, -1] == 1]
    orden = np.argsort(validas[:, 4])[::-1]
    mejor = validas[orden[0]]
    guardar_tabla("tabla_grilla_mejores", columnas[:7],
                  [[f"{f[0]:.2f}", f"{f[1]:.3f}", f"{f[2]:.3f}", f"{f[3]:.2f}", f"{f[4]:.4f}", f"{f[5]:.3f}",
                    f"{100 * f[6]:.2f}"] for f in validas[orden[:8]]],
                  ["$A$", r"$\beta$", r"$\gamma$", r"$\sigma$", "$S_P$", r"$\Delta t_{TP}$ [mes]", "Periodicidad [\\%]"])

    SP = grilla[:, 4].reshape([GRILLA[n] for n in RANGOS])
    resumen["grilla"] = {
        "combinaciones": len(grilla),
        "validas": len(validas),
        "peor_periodicidad": float(grilla[:, 6].max()),
        "mayor_valor_absoluto": float(grilla[:, 7].max()),
        "SP_min": float(validas[:, 4].min()),
        "desfase_min": float(validas[:, 5].min()),
        "desfase_max": float(validas[:, 5].max()),
        "casos_maxP_antes_de_maxT": int((grilla[:, 8] == 0).sum()),
        "mejor": dict(zip(columnas[:7], map(float, mejor[:7]))),
        "SP_creciente_en": {n: bool((np.diff(SP, axis=i) > 0).all()) for i, n in enumerate(RANGOS)},
        "SP_decreciente_en": {n: bool((np.diff(SP, axis=i) < 0).all()) for i, n in enumerate(RANGOS)},
    }
    return dict(zip(RANGOS, mejor[:4]))


def comparar(optimo, resumen):
    maximos = {nombre: RANGOS[nombre][1] for nombre in RANGOS}
    escenarios = {"Nominal": {n: NOMINAL[n] for n in RANGOS}, "Mayor $S_P$": optimo, "Todos al máximo": maximos}
    resultados = {nombre: evaluar(con_parametros(**p)) for nombre, p in escenarios.items()}

    filas = [[f"${SIMBOLO[n]}$"] + [f"{p[n]:.2f}" for p in escenarios.values()] for n in RANGOS]
    etiquetas = {"SP": "$S_P$", "desfase": r"$\Delta t_{TP}$ [mes]", "Pmax": r"$P_{\max}$ [u.p.]",
                 "Pmin": r"$P_{\min}$ [u.p.]", "Pmed": r"$P_{med}$ [u.p.]", "Tmin": r"$T_{\min}$ [°C]",
                 "Tmax": r"$T_{\max}$ [°C]", "Hmed": r"$H_{med}$ [u.h.]", "periodicidad": "Periodicidad [\\%]"}
    for clave, etiqueta in etiquetas.items():
        factor = 100 if clave == "periodicidad" else 1
        filas.append([etiqueta] + [f"{factor * r[clave]:.3f}" for r in resultados.values()])
    guardar_tabla("tabla_comparacion", ["", *escenarios], filas)
    resumen["escenarios"] = {nombre.replace("$", ""): {k: float(v) for k, v in r.items()}
                             for nombre, r in resultados.items()}

    nombres = ["T [°C]", "H [u.h.]", "P [u.p.]"]
    fig, ejes = plt.subplots(3, 2, figsize=(10, 6), sharex="col", sharey="row",
                             gridspec_kw={"width_ratios": [2, 1]})
    for nombre, estilo in zip(["Nominal", "Mayor $S_P$"], ["C0-", "C3-"]):
        t, y = euler(con_parametros(**escenarios[nombre]), N_PASOS)
        ciclo = t >= T_CICLO
        for i in range(3):
            ejes[i, 0].plot(t, y[:, i], estilo, lw=1, label=nombre)
            ejes[i, 1].plot(t[ciclo], y[ciclo, i], estilo, lw=1)
            ejes[i, 0].set_ylabel(nombres[i])
    ejes[0, 0].legend(fontsize=7)
    ejes[0, 0].set_title("Simulación completa")
    ejes[0, 1].set_title("Último ciclo anual")
    ejes[2, 0].set_xlabel("t [mes]")
    ejes[2, 1].set_xlabel("t [mes]")
    ejes[2, 0].set_xticks(range(0, 61, 6))
    ejes[2, 1].set_xticks(range(48, 61, 2))
    guardar_figura(fig, "fig_comparacion.png")


def main():
    resumen = {"h": 60 / N_PASOS, "sensibilidad": {}}
    escenario_nominal(resumen)
    sensibilidad(resumen)
    optimo = maximizar(resumen)
    comparar(optimo, resumen)
    (RESULTADOS / "parte2_resumen.json").write_text(json.dumps(resumen, indent=2))
    print(json.dumps(resumen, indent=2))


if __name__ == "__main__":
    main()
