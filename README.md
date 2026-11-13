# Modelación Numérica — TP 1

FIUBA · Curso Tarela · Segundo cuatrimestre de 2026

Andrés Ezequiel Arbesú · Padrón 102155

Modelo simplificado del fenómeno de El Niño. Resolución con Euler explícito,
análisis de convergencia, orden y estabilidad, y estudio de sensibilidad paramétrica.

## Archivos

- `src/`: código Python.
- `informe.pdf`: informe con los resultados.
- `codigo.txt`: copia del código fuente en un solo archivo, separado por nombre de archivo.

## Ejecución

Python 3.10 o superior.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

En Windows, activar el entorno con `.venv\Scripts\activate`.

La ejecución genera la carpeta `results/` con los gráficos, las tablas y los
resúmenes de las simulaciones. Cada parte también puede ejecutarse por separado:

```bash
python src/parte1.py
python src/parte2.py
```
