import numpy as np

T_FINAL = 60.0
Y0 = (1.0, 0.0, 0.0)
OMEGA = 2 * np.pi / 12

NOMINAL = {
    "Tmax": 4.0,
    "Roce": 0.30,
    "alpha": 0.05,
    "beta": 0.20,
    "gamma": 0.10,
    "delta": 0.40,
    "sigma": 0.70,
    "A": 0.45,
}


def con_parametros(**cambios):
    params = dict(NOMINAL)
    params.update(cambios)
    return params


def R(t, params):
    return params["Roce"] + params["A"] * np.cos(OMEGA * t - np.pi)


def f(t, y, params):
    T, H, P = y
    dT = R(t, params) * T * (1 - T / params["Tmax"]) - params["alpha"] * H
    dH = params["beta"] * T**2 - params["gamma"] * H * P
    dP = params["delta"] * H**2 - params["sigma"] * P
    return np.array([dT, dH, dP])


def euler(params, n_pasos, guardar_cada=1):
    h = T_FINAL / n_pasos
    forma = np.broadcast(*params.values()).shape
    y = np.empty((3,) + forma)
    y[0], y[1], y[2] = Y0

    assert n_pasos % guardar_cada == 0
    n_guardados = n_pasos // guardar_cada
    t_guardado = np.linspace(0, T_FINAL, n_guardados + 1)
    y_guardado = np.empty((n_guardados + 1,) + y.shape)
    y_guardado[0] = y

    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(n_pasos):
            y = y + h * f(n * h, y, params)
            if (n + 1) % guardar_cada == 0:
                y_guardado[(n + 1) // guardar_cada] = y
    return t_guardado, y_guardado
