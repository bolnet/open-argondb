"""Graph module — SmartGraph-like partitioning and parallel traversal."""

from open_arangodb.graph.manager import GraphManager
from open_arangodb.graph.parallel import ParallelTraverser

__all__ = ["GraphManager", "ParallelTraverser"]
