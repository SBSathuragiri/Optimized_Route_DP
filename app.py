import time
from flask import Flask, request, render_template
import networkx as nx
import folium
import requests
from haversine import haversine, Unit
import os
from dotenv import load_dotenv


app = Flask(__name__)
load_dotenv()

OVERPASS_URL = 'https://overpass-api.de/api/interpreter'
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")

ROAD_QUIETNESS = {
    'motorway': 5, 'trunk': 4, 'primary': 3, 'secondary': 2,
    'residential': 1, 'service': 1, 'footway': 0.5, 'cycleway': 0.5
}

# Fetch OSM data including footways & cycleways
def fetch_osm_data(south, west, north, east):
    query = f"""
    [out:json];
    (
      way["highway"]
      ( {south},{west},{north},{east} );
    );
    (._;>;);
    out body;
    """
    try:
        response = requests.get(OVERPASS_URL, params={'data': query})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching OSM data: {e}")
        return None


# Fetch real-time traffic data from TomTom
def get_real_time_traffic(start_coords, end_coords):
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{start_coords[0]},{start_coords[1]}:{end_coords[0]},{end_coords[1]}/json?key={TOMTOM_API_KEY}&traffic=true"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        print(f"🔹 TomTom Traffic Data: {data}")  # ✅ Print API response

        if 'routes' in data and len(data['routes']) > 0:
            return data['routes'][0]['summary']['travelTimeInSeconds'] / 60  # Convert to minutes
        return None
    except requests.exceptions.RequestException as e:
        print(f"⚠ Error fetching traffic data: {e}")
        return None

# Parse OSM data into a graph
def parse_osm_data(osm_data, path_type):
    graph = nx.Graph()
    nodes = {}

    for element in osm_data["elements"]:
        if element["type"] == "node":
            nodes[element["id"]] = (element["lat"], element["lon"])
        elif element["type"] == "way" and "nodes" in element:
            for i in range(len(element["nodes"]) - 1):
                node1, node2 = element["nodes"][i], element["nodes"][i + 1]
                if node1 in nodes and node2 in nodes:  # Ensure nodes exist
                    distance = haversine(nodes[node1], nodes[node2], unit=Unit.KILOMETERS)
                    speed = 40  # Default speed (km/h)

                    congestion_factor = 1  # Default to 1 for non-traffic
                    if path_type == "traffic":
                        congestion_factor = 1.5  # Assume moderate traffic
                        speed /= congestion_factor  

                    travel_time = (distance / speed) * 60  # Convert to minutes
                    graph.add_edge(node1, node2, weight=travel_time, traffic_factor=congestion_factor)

    print(f"✅ Graph Summary: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    return graph, nodes


def heuristic(n1, n2, nodes):
    """Haversine heuristic for A* algorithm."""
    return haversine(nodes[n1], nodes[n2], unit=Unit.KILOMETERS)


# Compute the best route
def get_routes(graph, nodes, start, waypoints, end, path_type, primary_algorithm="dijkstra"):
    try:
        start_node = min(nodes, key=lambda n: haversine(start, nodes[n]))
        end_node = min(nodes, key=lambda n: haversine(end, nodes[n]))

        if not nx.has_path(graph, start_node, end_node):
            print("⚠ Error: Start and End nodes are not connected.")
            return None, None

        if waypoints:
            waypoint_nodes = [min(nodes, key=lambda n: haversine(wp, nodes[n])) for wp in waypoints]
            all_nodes = [start_node] + waypoint_nodes + [end_node]
            path = []
            total_time = 0

            for i in range(len(all_nodes) - 1):
                if not nx.has_path(graph, all_nodes[i], all_nodes[i + 1]):
                    print(f"⚠ No path found between {all_nodes[i]} and {all_nodes[i + 1]}")
                    return None, None

                start_time = time.time()

                # Try primary algorithm first
                if primary_algorithm == "dijkstra":
                    try:
                        sub_path = nx.dijkstra_path(graph, all_nodes[i], all_nodes[i + 1], weight='weight')
                    except nx.NetworkXNoPath:
                        print("❌ Dijkstra failed, switching to A*")
                        sub_path = nx.astar_path(graph, all_nodes[i], all_nodes[i + 1], weight='weight',
                                                 heuristic=lambda n1, n2: heuristic(n1, n2, nodes))
                else:
                    try:
                        sub_path = nx.astar_path(graph, all_nodes[i], all_nodes[i + 1], weight='weight',
                                                 heuristic=lambda n1, n2: heuristic(n1, n2, nodes))
                    except nx.NetworkXNoPath:
                        print("❌ A* failed, switching to Dijkstra")
                        sub_path = nx.dijkstra_path(graph, all_nodes[i], all_nodes[i + 1], weight='weight')

                end_time = time.time()
                total_time += end_time - start_time
                path.extend(sub_path[:-1])  # Avoid duplicate nodes

            path.append(end_node)  # Add the final node
            return path, total_time
        else:
            start_time = time.time()

            # Try primary algorithm first
            try:
                if primary_algorithm == "dijkstra":
                    path = nx.dijkstra_path(graph, start_node, end_node, weight='weight')
                else:
                    path = nx.astar_path(graph, start_node, end_node, weight='weight',
                                         heuristic=lambda n1, n2: heuristic(n1, n2, nodes))
            except nx.NetworkXNoPath:
                print(f"❌ {primary_algorithm.capitalize()} failed, switching to the other algorithm.")
                if primary_algorithm == "dijkstra":
                    path = nx.astar_path(graph, start_node, end_node, weight='weight',
                                         heuristic=lambda n1, n2: heuristic(n1, n2, nodes))
                else:
                    path = nx.dijkstra_path(graph, start_node, end_node, weight='weight')

            end_time = time.time()
            total_time = end_time - start_time
            return path, total_time

    except nx.NetworkXNoPath:
        print("⚠ No path found between start and end.")
        return None, None
    except Exception as e:
        print(f"⚠ Error in get_routes(): {e}")
        return None, None


def estimate_travel_time(graph, path, nodes):
    """Estimate travel time using a default speed."""
    total_distance = sum(
        haversine(nodes[path[i]], nodes[path[i + 1]], unit=Unit.KILOMETERS)
        for i in range(len(path) - 1)
    )
    average_speed_kmh = 25
    extra_delay_per_node = 0.5 
    extra_delay = len(path) * extra_delay_per_node  

    return round((total_distance / average_speed_kmh) * 60 + extra_delay, 2) if total_distance else 0

def calculate_traffic_delay(path, graph):
    """Calculate traffic delay using real congestion factors."""
    delay = 0
    for i in range(len(path) - 1):
        edge_data = graph.get_edge_data(path[i], path[i + 1])
        if edge_data:
            congestion_factor = edge_data.get("traffic_factor", 1)
            delay += edge_data["weight"] * (congestion_factor - 1)
    return delay


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/show_route')
def show_route():
    # ✅ Step 1: Parse Request Parameters
    start_lat, start_lon = float(request.args.get('start_lat')), float(request.args.get('start_lon'))
    end_lat, end_lon = float(request.args.get('end_lat')), float(request.args.get('end_lon'))
    waypoints = request.args.get('waypoints', '')
    path_type = request.args.get('path_type', 'distance')

    waypoints_list = [tuple(map(float, wp.split(','))) for wp in waypoints.split(';')] if waypoints else []

    # ✅ Step 2: Fetch OSM Data
    osm_data = fetch_osm_data(start_lat - 1, start_lon - 1, end_lat + 1, end_lon + 1)
    print(f"Fetched OSM Data: {osm_data}")
    if not osm_data:
        print("⚠ Error: No OSM data found.")
        return "Error: No OSM data found."
    

    # ✅ Step 3: Parse Data & Create Graph
    graph, nodes = parse_osm_data(osm_data, path_type)
    print(f"Graph Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")

    if len(graph.nodes) == 0:
        print("⚠ Error: No roads found in the area.")
        return "Error: No roads found."

    # # ✅ Step 4: Find the Nearest Graph Nodes
    # dijkstra_path, dijkstra_time = get_routes(graph, nodes, (start_lat, start_lon), waypoints_list, (end_lat, end_lon), path_type, "dijkstra")
    # astar_path, astar_time = get_routes(graph, nodes, (start_lat, start_lon), waypoints_list, (end_lat, end_lon), path_type, "astar")

    # if not dijkstra_path or not astar_path:
    #     print("⚠ Error: No possible route found.")
    #     return "No possible route found."
    
    # dijkstra_travel_time = estimate_travel_time(graph, dijkstra_path, nodes)
    # astar_travel_time = estimate_travel_time(graph, astar_path, nodes)

    # # ✅ Compare results
    # print(f"🔹 Dijkstra: {len(dijkstra_path)} nodes, Execution Time: {dijkstra_time:.6f} sec, Travel Time: {dijkstra_travel_time:.2f} min")
    # print(f"🔹 A*: {len(astar_path)} nodes, Execution Time: {astar_time:.6f} sec, Travel Time: {astar_travel_time:.2f} min")

    # # ✅ Determine the best algorithm
    # best_algorithm = "A*" if astar_travel_time < dijkstra_travel_time else "Dijkstra"

    # # ✅ Step 5: Ensure Start & End Nodes Exist
    # if dijkstra_path[0] not in graph.nodes or dijkstra_path[-1] not in graph.nodes:
    #     print("⚠ Error: Start or End node not in graph!")
    #     return "No possible route found."

    # Try Dijkstra first, fall back to A* if necessary
    path, execution_time = get_routes(graph, nodes, (start_lat, start_lon), waypoints_list, (end_lat, end_lon), path_type, "dijkstra")

    if not path:
        print("❌ Both Dijkstra and A* failed to find a route.")
        return "No possible route found."
    
    travel_time = estimate_travel_time(graph, path, nodes)


    # ✅ Step 6: Assign Congestion Level
    congestion_color = "green"
    if travel_time  > 15:
        congestion_color = "red"
    elif travel_time  > 8:
        congestion_color = "orange"

    # ✅ Step 7: Generate Folium Map
    folium_map = folium.Map(location=[start_lat, start_lon], zoom_start=14)
    coords = [(nodes[node][0], nodes[node][1]) for node in path]
    folium.PolyLine(coords, color="blue", weight=4, opacity=0.8).add_to(folium_map)

    folium.Marker([start_lat, start_lon], popup="Start", icon=folium.Icon(color="green")).add_to(folium_map)
    for wp in waypoints_list:
        folium.Marker([wp[0], wp[1]], popup="Waypoint", icon=folium.Icon(color="orange")).add_to(folium_map)
    folium.Marker([end_lat, end_lon], popup="End", icon=folium.Icon(color="red")).add_to(folium_map)

    # ✅ Render Template with All Required Information
    return render_template('route.html',
                           map=folium_map._repr_html_(),
                           travel_time=travel_time,
                           congestion_color=congestion_color,
                           path_type=path_type,
                           num_nodes=len(path),
                           execution_time=execution_time)


if __name__ == '__main__':
    # app.run(debug=True)
    port = int(os.environ.get('PORT', 5000))  
    app.run(host='0.0.0.0', port=port, debug=True)
