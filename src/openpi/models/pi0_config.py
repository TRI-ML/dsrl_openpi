# Re-export Pi0Config from pi0.py so that imports from either location
# resolve to the same class (fixes isinstance checks in ModelTransformFactory).
from openpi.models.pi0 import Pi0Config

__all__ = ["Pi0Config"]
