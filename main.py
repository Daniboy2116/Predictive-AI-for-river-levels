"""Pipeline completo: genera datos (si faltan), entrena y evalúa todos los modelos.

Uso:
    python main.py                      # todos los modelos
    python main.py --models tree arima  # subconjunto
"""
from __future__ import annotations

import argparse
import importlib
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GROUPS = {
    "arima": "src.models.arima_model",
    "tree": "src.models.tree_models",
    "neural": "src.models.neural_models",
    "google": "src.models.google_model",
}


def ensure_data() -> None:
    """Genera el dataset simulado si no existe el Excel."""
    if not (ROOT / "data" / "datos_rios.xlsx").exists():
        import create_dummy_data
        create_dummy_data.main()


def main() -> int:
    """Ejecuta secuencialmente cada grupo de modelos; un fallo no detiene al resto."""
    parser = argparse.ArgumentParser(description="Entrenamiento y evaluación de modelos")
    parser.add_argument("--models", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    args = parser.parse_args()

    ensure_data()
    failures: list[str] = []
    for group in args.models:
        print(f"\n=== {group.upper()} ===", flush=True)
        t0 = time.perf_counter()
        try:
            importlib.import_module(GROUPS[group]).run()  # importación perezosa
            print(f"--- {group} OK ({time.perf_counter() - t0:.1f}s)")
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            failures.append(group)

    from src.evaluate import load_metrics
    table = load_metrics().sort_values(["Estacion", "RMSE"])
    print("\n=== RESUMEN (ordenado por RMSE) ===")
    print(table.to_string(index=False, float_format="%.4f"))
    if failures:
        print(f"\nGrupos con errores: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
