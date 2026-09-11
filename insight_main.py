"""CLI for the V2 Hermes insight layer.

Forecast generation remains `python main.py`; this command consumes the
resulting V2 diagnostic view and persists Region/GM/CEO narratives.
"""
from insight.batch import main


if __name__ == "__main__":
    main()
