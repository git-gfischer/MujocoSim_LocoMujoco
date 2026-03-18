from .base import Terrain
from .static import StaticTerrain
from .dynamic import DynamicTerrain
from .rough import RoughTerrain
from .flat_floor import FlatFloorTerrain

# register all terrains
StaticTerrain.register()
RoughTerrain.register()
FlatFloorTerrain.register()
