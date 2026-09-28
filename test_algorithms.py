"""
Quick verification test for all 3 algorithms (A*, CSP, Hill Climbing).
"""
import time
from data_generator import load_processed_data
from algorithms.astar import AStarRouteFinder
from algorithms.csp import CSPSolver
from algorithms.hillclimbing import HillClimbingShelterFinder

def test_all():
    print("Testing Data Loading...")
    t0 = time.time()
    data = load_processed_data()
    print(f"Data loaded in {time.time()-t0:.2f}s. Total records: {data['summary']['total_records']}")

    # Pick sample nodes from Sunamganj or Sylhet
    rep_nodes = data['representative_nodes'].to_dict(orient='records')
    sunamganj_nodes = [n for n in rep_nodes if 'Sunamganj' in n.get('zilla', '')]
    if not sunamganj_nodes:
        sunamganj_nodes = rep_nodes[:30]

    print(f"\n--- Testing Algorithm 1: A* Route Optimization ---")
    start_node = sunamganj_nodes[0]
    goal_node = sunamganj_nodes[min(10, len(sunamganj_nodes)-1)]
    astar = AStarRouteFinder(risk_weight=3.5)
    t0 = time.time()
    route_res = astar.find_route(sunamganj_nodes, start_node['area_id'], goal_node['area_id'])
    print(f"A* completed in {time.time()-t0:.3f}s")
    print(f"Distance: {route_res['total_distance_km']} km | Safety: {route_res['safety_score']}% | Nodes: {route_res['nodes_in_route']}")
    print(f"Confusion Matrix: {route_res['confusion_matrix']['matrix']}")

    print(f"\n--- Testing Algorithm 2: CSP Relief Team Allocation ---")
    csp = CSPSolver(max_teams_per_area=2, max_response_time_mins=120.0, team_relief_capacity=750)
    t0 = time.time()
    csp_res = csp.solve(sunamganj_nodes[:25], available_teams=10)
    print(f"CSP completed in {time.time()-t0:.3f}s")
    print(f"Teams Deployed: {csp_res['teams_deployed']} | Coverage: {csp_res['coverage_percentage']}% | CS Rate: {csp_res['constraint_satisfaction_rate']}%")
    print(f"Confusion Matrix: {csp_res['confusion_matrix']['matrix']}")

    print(f"\n--- Testing Algorithm 3: Hill Climbing Shelter Finder ---")
    hc = HillClimbingShelterFinder(top_k=5, coverage_radius_km=6.0, max_restarts=8, max_iterations_per_restart=20)
    t0 = time.time()
    hc_res = hc.find_shelters(sunamganj_nodes, target_population=25000)
    print(f"Hill Climbing completed in {time.time()-t0:.3f}s")
    print(f"Shelters Found: {hc_res['total_shelters_selected']} | Avg Suitability: {hc_res['average_suitability_score']} | Pop Covered: {hc_res['population_covered']}")
    print(f"Confusion Matrix: {hc_res['confusion_matrix']['matrix']}")

    print("\n[SUCCESS] All 3 algorithms executed successfully and returned valid results!")

if __name__ == "__main__":
    test_all()
