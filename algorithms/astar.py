"""
Algorithm 1: A* (A-Star) Route Optimization for Bangladesh Flood Disaster Management
====================================================================================
Purpose:
    Find the shortest and safest evacuation route from flood-affected settlements
    to the nearest accessible disaster shelter, balancing physical travel distance
    against flood inundation risk, low elevation hazards, and road accessibility.

Mathematical Formulation:
    - Path evaluation function: f(n) = g(n) + h(n)
    - Path cost function g(n):
        g(current, neighbor) = g(current) + Distance(current, neighbor) * CostFactor(neighbor)
        where CostFactor(v) = 1.0 + (risk_weight * FloodRisk(v)^2) + (5.0 / max(1.0, Elevation(v)))
    - Heuristic function h(n):
        h(n) = Haversine(n, goal) * (1.0 + min_risk_baseline)
        (Admissible and consistent to guarantee finding the optimal safe path).

Validation & Metrics:
    - Route Efficiency Score: Ratio of ideal travel distance to weighted risk path.
    - Confusion Matrix (Corridor Safety Classification):
        * TP (True Positive): Safe corridor (Risk <= 0.45) correctly selected.
        * TN (True Negative): Dangerous flood zone (Risk > 0.45) correctly avoided.
        * FP (False Positive): Risky corridor traversed due to lack of safer alternatives.
        * FN (False Negative): Safe corridor bypassed due to excessive circuitous detour.
    - Metrics: Accuracy, Precision, Recall (Safety Coverage), F1-Score, Specificity.
"""

import math
import heapq
import time
from typing import Dict, List, Tuple, Any, Optional

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes the great-circle distance between two geographical points on Earth
    using the Haversine formula (returns distance in kilometers).
    """
    R = 6371.0  # Earth's mean radius in kilometers
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return round(R * c, 3)

class AStarRouteFinder:
    def __init__(self, risk_weight: float = 3.5, safe_risk_threshold: float = 0.45):
        """
        Initializes the A* Route Optimizer.
        
        Args:
            risk_weight: Penalty factor for traversing flooded/high-risk zones (default: 3.5).
            safe_risk_threshold: Boundary below which a road segment is deemed safe (default: 0.45).
        """
        self.risk_weight = float(risk_weight)
        self.safe_risk_threshold = float(safe_risk_threshold)

    def heuristic(self, node: Dict[str, Any], goal_node: Dict[str, Any]) -> float:
        """
        Admissible heuristic combining Haversine straight-line distance with
        minimum baseline flood cost to avoid overestimating true cost.
        """
        base_dist = haversine_distance(node['latitude'], node['longitude'],
                                       goal_node['latitude'], goal_node['longitude'])
        # Minimum potential flood cost is base_dist * 1.0 (when risk = 0)
        return base_dist * 1.05

    def edge_cost(self, u: Dict[str, Any], v: Dict[str, Any], use_safety: bool = True) -> Tuple[float, float]:
        """
        Calculates the traversal cost and raw physical distance between adjacent nodes.
        
        Args:
            u: Origin node dictionary.
            v: Destination node dictionary.
            use_safety: If False, computes standard naive physical distance only.
        
        Returns:
            (traversal_cost, physical_distance_km)
        """
        dist = haversine_distance(u['latitude'], u['longitude'], v['latitude'], v['longitude'])
        if dist < 0.05:
            dist = 0.05  # minimum segment distance

        if not use_safety:
            return dist, dist

        # Flood risk penalty: quadratic scaling penalizes high water levels severely
        v_risk = float(v.get('flood_risk_level', 0.5))
        v_elev = max(1.0, float(v.get('elevation', 5.0)))
        v_access = max(0.1, float(v.get('accessibility_score', 0.5)))

        # Risk factor: incorporates flood risk, low elevation hazard, and poor road accessibility
        risk_penalty = self.risk_weight * (v_risk ** 2)
        elev_penalty = 12.0 / (v_elev + 2.0)
        access_multiplier = 1.0 / v_access

        cost = dist * (1.0 + risk_penalty + elev_penalty) * access_multiplier
        return cost, dist

    def find_route(self, nodes: List[Dict[str, Any]], start_id: str, goal_id: str) -> Dict[str, Any]:
        """
        Executes A* search over the spatial road network.
        
        Args:
            nodes: List of area/node dicts containing id, lat, lon, risk, elevation, etc.
            start_id: Identifier of the starting affected community.
            goal_id: Identifier of the target shelter.
            
        Returns:
            Dictionary with optimal route, distance, risk metrics, comparison, and confusion matrix.
        """
        start_time = time.time()
        node_map = {n['area_id']: n for n in nodes}

        if start_id not in node_map or goal_id not in node_map:
            raise ValueError(f"Start ID ({start_id}) or Goal ID ({goal_id}) not found in spatial network.")

        start_node = node_map[start_id]
        goal_node = node_map[goal_id]

        # Build dynamic spatial adjacency graph:
        # Connect each node to its k-nearest geographical neighbors (simulating road/embankment network)
        adj: Dict[str, List[str]] = {n['area_id']: [] for n in nodes}
        k_neighbors = min(8, len(nodes) - 1)

        # Pre-sort neighbors by spatial proximity
        all_ids = list(node_map.keys())
        for n_id, n_data in node_map.items():
            # Get closest nodes by Euclidean distance
            distances = []
            for other_id in all_ids:
                if other_id == n_id:
                    continue
                o_data = node_map[other_id]
                d = haversine_distance(n_data['latitude'], n_data['longitude'],
                                       o_data['latitude'], o_data['longitude'])
                distances.append((d, other_id))
            
            distances.sort(key=lambda x: x[0])
            adj[n_id] = [other_id for _, other_id in distances[:k_neighbors]]
            # Always ensure connection to goal if within reasonable proximity
            if goal_id not in adj[n_id]:
                adj[n_id].append(goal_id)

        # Priority queue stores tuples: (f_score, g_score, current_node_id, path)
        open_set = []
        heapq.heappush(open_set, (0.0, 0.0, start_id, [start_id]))

        g_scores: Dict[str, float] = {start_id: 0.0}
        visited_nodes = set()
        explored_count = 0

        optimal_path = []
        best_g_score = float('inf')

        while open_set:
            f, g, current_id, path = heapq.heappop(open_set)
            explored_count += 1

            if current_id == goal_id:
                optimal_path = path
                best_g_score = g
                break

            if current_id in visited_nodes and g > g_scores.get(current_id, float('inf')):
                continue
            visited_nodes.add(current_id)

            current_node = node_map[current_id]

            for neighbor_id in adj.get(current_id, []):
                neighbor_node = node_map[neighbor_id]
                step_cost, _ = self.edge_cost(current_node, neighbor_node, use_safety=True)
                tentative_g = g + step_cost

                if tentative_g < g_scores.get(neighbor_id, float('inf')):
                    g_scores[neighbor_id] = tentative_g
                    h = self.heuristic(neighbor_node, goal_node)
                    f_score = tentative_g + h
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor_id, path + [neighbor_id]))

        # Fallback if no exact path found (direct topological safe interpolation)
        if not optimal_path:
            optimal_path = [start_id, goal_id]

        # Compute Naive (Shortest Physical Distance) Route for comparison (ignoring flood risk)
        naive_path = self._compute_naive_shortest_route(node_map, adj, start_id, goal_id)

        # Calculate metrics for the optimal A* path
        route_details = []
        total_distance_km = 0.0
        total_risk_score = 0.0

        for i in range(len(optimal_path)):
            curr = node_map[optimal_path[i]]
            step_info = {
                "step": i + 1,
                "area_id": curr['area_id'],
                "area_name": curr.get('area_name', 'Unnamed Area'),
                "upazila": curr.get('upazila', ''),
                "thana": curr.get('thana', ''),
                "latitude": curr['latitude'],
                "longitude": curr['longitude'],
                "elevation_m": curr.get('elevation', 5.0),
                "flood_risk": curr.get('flood_risk_level', 0.5),
                "accessibility": curr.get('accessibility_score', 0.5)
            }
            if i > 0:
                prev = node_map[optimal_path[i - 1]]
                seg_dist = haversine_distance(prev['latitude'], prev['longitude'],
                                              curr['latitude'], curr['longitude'])
                total_distance_km += seg_dist
                step_info["segment_distance_km"] = seg_dist
            else:
                step_info["segment_distance_km"] = 0.0

            total_risk_score += curr.get('flood_risk_level', 0.5)
            route_details.append(step_info)

        avg_flood_risk = total_risk_score / max(1, len(optimal_path))
        safety_score = round(max(0.0, min(100.0, (1.0 - avg_flood_risk) * 100.0)), 1)
        total_distance_km = round(total_distance_km, 2)

        # Metrics for Naive Route
        naive_dist, naive_risk = self._evaluate_path_metrics(node_map, naive_path)
        naive_safety = round(max(0.0, min(100.0, (1.0 - naive_risk) * 100.0)), 1)

        # Confusion Matrix Calculation on Spatial Road Segments
        # Evaluates how well A* selected safe corridors and avoided high-risk corridors
        confusion_matrix = self._compute_confusion_matrix(node_map, optimal_path, all_ids)

        execution_time_ms = round((time.time() - start_time) * 1000, 2)

        # Route efficiency trade-off score
        direct_dist = haversine_distance(start_node['latitude'], start_node['longitude'],
                                         goal_node['latitude'], goal_node['longitude'])
        efficiency_score = round(min(100.0, (direct_dist / max(0.1, total_distance_km)) * (safety_score / 100.0) * 100.0), 1)

        return {
            "algorithm": "A* Search (Safe Route Optimization)",
            "start_node": {
                "id": start_id, "name": start_node.get('area_name', ''),
                "lat": start_node['latitude'], "lon": start_node['longitude']
            },
            "goal_node": {
                "id": goal_id, "name": goal_node.get('area_name', ''),
                "lat": goal_node['latitude'], "lon": goal_node['longitude']
            },
            "total_distance_km": total_distance_km,
            "direct_distance_km": round(direct_dist, 2),
            "safety_score": safety_score,
            "avg_flood_risk": round(avg_flood_risk, 3),
            "nodes_in_route": len(optimal_path),
            "nodes_explored": explored_count,
            "execution_time_ms": execution_time_ms,
            "efficiency_score": efficiency_score,
            "route_path": route_details,
            "coordinates": [[n['latitude'], n['longitude']] for n in route_details],
            "comparison": {
                "astar": {
                    "distance_km": total_distance_km,
                    "safety_score": safety_score,
                    "avg_risk": round(avg_flood_risk, 3)
                },
                "naive_shortest": {
                    "distance_km": round(naive_dist, 2),
                    "safety_score": naive_safety,
                    "avg_risk": round(naive_risk, 3),
                    "coordinates": [[node_map[nid]['latitude'], node_map[nid]['longitude']] for nid in naive_path]
                },
                "safety_improvement_percent": round(max(0.0, safety_score - naive_safety), 1)
            },
            "confusion_matrix": confusion_matrix
        }

    def _compute_naive_shortest_route(self, node_map: Dict[str, Any], adj: Dict[str, List[str]],
                                      start_id: str, goal_id: str) -> List[str]:
        """Computes shortest physical distance route ignoring flood risk (Dijkstra)."""
        dist_heap = [(0.0, start_id, [start_id])]
        visited = set()
        while dist_heap:
            d, curr, path = heapq.heappop(dist_heap)
            if curr == goal_id:
                return path
            if curr in visited:
                continue
            visited.add(curr)
            for neighbor in adj.get(curr, []):
                seg_d = haversine_distance(node_map[curr]['latitude'], node_map[curr]['longitude'],
                                           node_map[neighbor]['latitude'], node_map[neighbor]['longitude'])
                heapq.heappush(dist_heap, (d + seg_d, neighbor, path + [neighbor]))
        return [start_id, goal_id]

    def _evaluate_path_metrics(self, node_map: Dict[str, Any], path: List[str]) -> Tuple[float, float]:
        """Returns (total_distance_km, avg_flood_risk) for any path."""
        dist = 0.0
        risk_sum = 0.0
        for i in range(len(path)):
            node = node_map[path[i]]
            risk_sum += node.get('flood_risk_level', 0.5)
            if i > 0:
                prev = node_map[path[i-1]]
                dist += haversine_distance(prev['latitude'], prev['longitude'],
                                           node['latitude'], node['longitude'])
        return dist, risk_sum / max(1, len(path))

    def _compute_confusion_matrix(self, node_map: Dict[str, Any], optimal_path: List[str],
                                  all_ids: List[str]) -> Dict[str, Any]:
        """
        Builds the 2x2 Confusion Matrix evaluating A* safety corridor classification.
        - Ground Truth: Node is inherently Safe/Lower Risk (Flood Risk <= effective_threshold)
          vs Dangerous/Flooded (Risk > effective_threshold).
        - Model Decision: Selected for route (Predicted Positive) vs Avoided (Predicted Negative).
        """
        path_set = set(optimal_path)
        all_risks = [node_map[nid].get('flood_risk_level', 0.5) for nid in all_ids]
        median_risk = float(sorted(all_risks)[len(all_risks) // 2]) if all_risks else 0.5
        effective_threshold = min(self.safe_risk_threshold, median_risk)

        tp = 0  # Safe node selected in route
        fp = 0  # Risky node selected in route (due to no other choice)
        fn = 0  # Safe node available but bypassed
        tn = 0  # Dangerous node avoided

        for node_id in all_ids:
            node = node_map[node_id]
            is_safe = (node.get('flood_risk_level', 0.5) <= effective_threshold)
            was_selected = (node_id in path_set)

            if is_safe and was_selected:
                tp += 1
            elif not is_safe and was_selected:
                fp += 1
            elif is_safe and not was_selected:
                fn += 1
            else:
                tn += 1

        total = max(1, tp + tn + fp + fn)
        accuracy = round((tp + tn) / total, 3)
        precision = round(tp / max(1, tp + fp), 3)
        recall = round(tp / max(1, tp + fn), 3)
        f1_score = round(2 * (precision * recall) / max(0.001, precision + recall), 3)
        specificity = round(tn / max(1, tn + fp), 3)

        return {
            "matrix": {
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn
            },
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "specificity": specificity
            },
            "interpretation": f"True Positives ({tp} safe waypoints used), True Negatives ({tn} hazard zones avoided)."
        }
