"""Multi-view episodic image cache for the recall tool."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from agenticnav_paper_replica.geometry.pinhole import VIEW_AZIMUTHS_DEG, VIEW_LABELS


@dataclass
class RecallResult:
    success: bool
    image: Optional[np.ndarray] = None
    error_message: Optional[str] = None
    node: int = 0
    view: int = 0


class EpisodicImageCache:
    """Stores multi-view RGB observations indexed by decision node t and view k."""

    def __init__(self):
        # Dict mapping node_idx -> Dict[view_idx, np.ndarray (H, W, 3) uint8]
        self._cache: Dict[int, Dict[int, np.ndarray]] = {}
        self.current_node: int = 1

    def set_current_node(self, node: int) -> None:
        self.current_node = node

    def store_node_observations(
        self,
        node: int,
        rgb_views: Dict[int, np.ndarray],
    ) -> None:
        """Store 4 views for a given node.

        rgb_views maps view_idx (0, 1, 2, 3) to (H, W, 3) uint8 arrays.
        """
        self._cache[node] = {}
        for v_idx, img in rgb_views.items():
            if isinstance(img, Image.Image):
                self._cache[node][v_idx] = np.array(img, dtype=np.uint8)
            else:
                self._cache[node][v_idx] = img.astype(np.uint8, copy=True)

    def recall(self, node: int, view: int) -> RecallResult:
        """Recall cached RGB view for a past decision node.

        Constraints:
            - node must be in [1, current_node - 1]
            - view must be in {0, 1, 2, 3}
        """
        if node >= self.current_node:
            return RecallResult(
                success=False,
                error_message=f"Invalid node={node}. Recall is only allowed for past nodes (1 <= node <= {self.current_node - 1}). Current node is {self.current_node}.",
                node=node,
                view=view,
            )

        if node < 1:
            return RecallResult(
                success=False,
                error_message=f"Invalid node={node}. Nodes are 1-indexed (1 <= node <= {self.current_node - 1}).",
                node=node,
                view=view,
            )

        if view not in VIEW_AZIMUTHS_DEG:
            return RecallResult(
                success=False,
                error_message=f"Invalid view={view}. Must be one of {list(VIEW_AZIMUTHS_DEG.keys())} ({VIEW_LABELS}).",
                node=node,
                view=view,
            )

        if node not in self._cache or view not in self._cache[node]:
            return RecallResult(
                success=False,
                error_message=f"No cached observation found for node={node}, view={view}.",
                node=node,
                view=view,
            )

        return RecallResult(
            success=True,
            image=self._cache[node][view],
            node=node,
            view=view,
        )

    def get_all_cached_nodes(self) -> List[int]:
        return sorted(list(self._cache.keys()))

    def clear(self) -> None:
        self._cache.clear()
        self.current_node = 1

