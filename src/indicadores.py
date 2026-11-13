import numpy as np

T_CICLO = 48.0
PERIODO = 12.0


def integral_trapecios(t, y):
    return np.sum((y[1:] + y[:-1]) * np.diff(t).reshape((-1,) + (1,) * (y.ndim - 1)), axis=0) / 2


def indicadores(t, y):
    ciclo = t >= T_CICLO - 1e-9
    tc = t[ciclo]
    T = y[ciclo, 0]
    P = y[ciclo, 2]

    Pmax = P.max(axis=0)
    Pmin = P.min(axis=0)
    Pmed = integral_trapecios(tc, P) / (tc[-1] - tc[0])

    t_Tmax = tc[T.argmax(axis=0)]
    t_Pmax = tc[P.argmax(axis=0)]
    desfase = (t_Pmax - t_Tmax) % PERIODO

    return {
        "SP": (Pmax - Pmin) / Pmed,
        "desfase": desfase,
        "Pmax": Pmax,
        "Pmin": Pmin,
        "Pmed": Pmed,
        "t_Tmax": t_Tmax,
        "t_Pmax": t_Pmax,
    }


def error_periodicidad(t, y):
    k = int(round(PERIODO / (t[1] - t[0])))
    ultimo = y[-k - 1:]
    anterior = y[-2 * k - 1:-k]
    amplitud = ultimo.max(axis=0) - ultimo.min(axis=0)
    diferencia = np.abs(ultimo - anterior).max(axis=0)
    return (diferencia / amplitud).max(axis=0)
