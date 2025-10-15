#include <iostream>
/*
Student number: 24871125
Name: Brian Kodibona
Project: Smart City Route Management System
Single-file submission with ASCII visualization and full XAI comments.
Compile: g++ -std=c++17 24871125_FP.cpp -o route_manager
Run: ./route_manager
*/

#include <bits/stdc++.h>
using namespace std;

/* -------------------------
   Data structures & types
   ------------------------- */

struct Edge {
    int to;                 // destination node id
    double distance_km;     // distance in km
    double travel_time;     // expected minutes
    double cost;            // monetary cost
    string name;            // route name/identifier
    int id;                 // unique edge id
};

struct Node {
    string name;
    int x = -1, y = -1;     // optional coordinates for ASCII viz
};

class Graph {
private:
    vector<Node> nodes;                 // node id -> Node
    vector<list<Edge>> adj;             // adjacency list
    unordered_map<string,int> name_to_id;// mapping from node name to id
    set<string> route_names;            // set of unique route names
    int next_edge_id = 1;

    // Undo/Redo stacks (store simple snapshots)
    struct Action {
        string type; // "add", "remove", "update"
        int from;
        Edge snapshot;
    };
    stack<Action> undo_stack;
    stack<Action> redo_stack;

public:
    Graph() {}

    // get existing node id or create new
    int get_or_create_node(const string& name) {
        auto it = name_to_id.find(name);
        if (it != name_to_id.end()) return it->second;
        int id = nodes.size();
        nodes.push_back({name, -1, -1});
        adj.emplace_back();
        name_to_id[name] = id;
        return id;
    }

    int find_node(const string& name) const {
        auto it = name_to_id.find(name);
        if (it == name_to_id.end()) return -1;
        return it->second;
    }

    // add route (directed)
    bool add_route(const string& from, const string& to, double distance_km, double travel_time, double cost, const string& routeName) {
        if (route_names.count(routeName)) {
            cout << "Route name already exists (must be unique). Use a different name.\n";
            return false;
        }
        int u = get_or_create_node(from);
        int v = get_or_create_node(to);
        Edge e{v, distance_km, travel_time, cost, routeName, next_edge_id++};
        adj[u].push_back(e);
        route_names.insert(routeName);

        // Save for undo
        undo_stack.push({"add", u, e});
        // clear redo stack
        while(!redo_stack.empty()) redo_stack.pop();

        cout << "Added route: " << from << " -> " << to << " (" << routeName << ")\n";
        return true;
    }

    // remove route by routeName (origin specified)
    bool remove_route(const string& from, const string& routeName) {
        int u = find_node(from);
        if (u == -1) return false;
        for (auto it = adj[u].begin(); it != adj[u].end(); ++it) {
            if (it->name == routeName) {
                Edge snapshot = *it;
                // remove
                adj[u].erase(it);
                route_names.erase(routeName);
                undo_stack.push({"remove", u, snapshot});
                while(!redo_stack.empty()) redo_stack.pop();
                cout << "Removed route " << routeName << " from " << from << "\n";
                return true;
            }
        }
        return false;
    }

    // update route attributes
    bool update_route(const string& from, const string& routeName, double new_dist, double new_time, double new_cost) {
        int u = find_node(from);
        if (u == -1) return false;
        for (auto &e : adj[u]) {
            if (e.name == routeName) {
                Edge old = e;
                undo_stack.push({"update", u, old});
                e.distance_km = new_dist;
                e.travel_time = new_time;
                e.cost = new_cost;
                while(!redo_stack.empty()) redo_stack.pop();
                cout << "Updated route " << routeName << " from " << from << "\n";
                return true;
            }
        }
        return false;
    }

    // Undo last structural action
    bool undo() {
        if (undo_stack.empty()) {
            cout << "Nothing to undo.\n";
            return false;
        }
        Action a = undo_stack.top(); undo_stack.pop();
        if (a.type == "add") {
            // remove edge we added
            int u = a.from;
            int eid = a.snapshot.id;
            for (auto it = adj[u].begin(); it != adj[u].end(); ++it) {
                if (it->id == eid) {
                    redo_stack.push(a);
                    route_names.erase(it->name);
                    adj[u].erase(it);
                    cout << "Undo: removed edge id " << eid << "\n";
                    return true;
                }
            }
        } else if (a.type == "remove") {
            // re-add snapshot
            int u = a.from;
            adj[u].push_back(a.snapshot);
            route_names.insert(a.snapshot.name);
            redo_stack.push(a);
            cout << "Undo: re-added route " << a.snapshot.name << "\n";
            return true;
        } else if (a.type == "update") {
            int u = a.from;
            int eid = a.snapshot.id;
            for (auto &e : adj[u]) {
                if (e.id == eid) {
                    Edge current = e;
                    e = a.snapshot;
                    redo_stack.push({"update", u, current});
                    cout << "Undo: reverted update on route " << e.name << "\n";
                    return true;
                }
            }
        }
        return false;
    }

    // Redo
    bool redo() {
        if (redo_stack.empty()) {
            cout << "Nothing to redo.\n";
            return false;
        }
        Action a = redo_stack.top(); redo_stack.pop();
        if (a.type == "add") {
            adj[a.from].push_back(a.snapshot);
            route_names.insert(a.snapshot.name);
            undo_stack.push(a);
            cout << "Redo: re-added route " << a.snapshot.name << "\n";
            return true;
        } else if (a.type == "remove") {
            int u = a.from; int eid = a.snapshot.id;
            for (auto it = adj[u].begin(); it != adj[u].end(); ++it) {
                if (it->id == eid) {
                    route_names.erase(it->name);
                    adj[u].erase(it);
                    undo_stack.push(a);
                    cout << "Redo: removed route id " << eid << "\n";
                    return true;
                }
            }
            return false;
        } else if (a.type == "update") {
            int u = a.from; int eid = a.snapshot.id;
            for (auto &e : adj[u]) {
                if (e.id == eid) {
                    Edge old = e;
                    e = a.snapshot;
                    undo_stack.push({"update", u, old});
                    cout << "Redo: reapplied update on route " << e.name << "\n";
                    return true;
                }
            }
            return false;
        }
        return false;
    }

    // List all routes
    void print_all_routes() const {
        cout << "\n----- All routes -----\n";
        if (nodes.empty()) {
            cout << "No nodes/routes present.\n";
            return;
        }
        for (int u = 0; u < (int)nodes.size(); ++u) {
            if (adj[u].empty()) continue;
            cout << nodes[u].name << ":\n";
            for (const auto &e : adj[u]) {
                cout << "  [" << e.id << "] " << e.name << " -> " << nodes[e.to].name
                     << " | dist: " << e.distance_km << " km"
                     << " | time: " << e.travel_time << " min"
                     << " | cost: " << e.cost << "\n";
            }
        }
        cout << "----------------------\n";
    }

    // Helper: gather edges in a vector
    vector<pair<pair<int,int>, Edge>> list_all_edges() const {
        vector<pair<pair<int,int>, Edge>> out;
        for (int u = 0; u < (int)adj.size(); ++u) {
            for (const auto &e : adj[u]) out.push_back({{u, e.to}, e});
        }
        return out;
    }

    // Sort routes by chosen metric: 0=distance,1=time,2=cost
    void sort_and_print_routes(int metric) const {
        auto edges = list_all_edges();
        // Functor comparator example
        struct Comparator {
            int m;
            Comparator(int mm): m(mm) {}
            bool operator()(const pair<pair<int,int>, Edge>& a, const pair<pair<int,int>, Edge>& b) const {
                double va = (m==0? a.second.distance_km : (m==1? a.second.travel_time : a.second.cost));
                double vb = (m==0? b.second.distance_km : (m==1? b.second.travel_time : b.second.cost));
                return va < vb;
            }
        } comp(metric);

        sort(edges.begin(), edges.end(), comp);

        cout << "\nRoutes sorted by " << (metric==0?"distance(km)":(metric==1?"travel_time(min)":"cost")) << " (ascending):\n";
        // XAI: Routes are sorted using a custom functor to show how to compare by different metrics.
        for (const auto &p : edges) {
            cout << "[" << p.second.id << "] " << p.second.name << " : " << nodes[p.first.first].name
                 << " -> " << nodes[p.first.second].name << " | ";
            if (metric==0) cout << p.second.distance_km << " km\n";
            else if (metric==1) cout << p.second.travel_time << " min\n";
            else cout << p.second.cost << " units\n";
        }
    }

    // Dijkstra: weightChoice 0=distance,1=time,2=cost
    pair<vector<double>, vector<int>> dijkstra(int src, int weightChoice) const {
        int n = nodes.size();
        const double INF = 1e18;
        vector<double> dist(n, INF);
        vector<int> prev(n, -1);
        vector<char> vis(n, false);
        using P = pair<double,int>;
        priority_queue<P, vector<P>, greater<P>> pq;
        dist[src] = 0;
        pq.push({0, src});
        while (!pq.empty()) {
            auto [d,u] = pq.top(); pq.pop();
            if (vis[u]) continue;
            vis[u] = 1;
            // XAI: Node u chosen next because it currently has the smallest known distance 'd' among all unreached nodes.
            for (const auto &e : adj[u]) {
                double w = (weightChoice==0? e.distance_km : (weightChoice==1? e.travel_time : e.cost));
                if (d + w < dist[e.to]) {
                    // XAI: Updating node e.to because path via u yields smaller cost than previously known.
                    dist[e.to] = d + w;
                    prev[e.to] = u;
                    pq.push({dist[e.to], e.to});
                }
            }
        }
        return {dist, prev};
    }

    // Reconstruct path
    vector<int> reconstruct_path(int src, int dest, const vector<int>& prev) const {
        vector<int> path;
        int cur = dest;
        while (cur != -1) {
            path.push_back(cur);
            if (cur == src) break;
            cur = prev[cur];
        }
        reverse(path.begin(), path.end());
        if (path.empty() || path.front() != src) return {};
        return path;
    }

    // BFS hop-count path
    vector<int> bfs_path(int src, int dest) const {
        int n = nodes.size();
        vector<int> prev(n,-1);
        vector<char> vis(n,0);
        queue<int> q;
        q.push(src); vis[src]=1;
        while(!q.empty()) {
            int u = q.front(); q.pop();
            if (u==dest) break;
            for (const auto &e : adj[u]) {
                if (!vis[e.to]) {
                    vis[e.to]=1;
                    prev[e.to]=u;
                    q.push(e.to);
                }
            }
        }
        return reconstruct_path(src, dest, prev);
    }

    // compute metric for a path
    double compute_metric_for_path(const vector<int>& path, int weightChoice) const {
        double total = 0;
        for (size_t i=0;i+1<path.size();++i) {
            int u = path[i], v = path[i+1];
            double best = 1e18;
            for (const auto &e: adj[u]) {
                if (e.to == v) {
                    double w = (weightChoice==0? e.distance_km : (weightChoice==1? e.travel_time : e.cost));
                    if (w < best) best = w;
                }
            }
            if (best >= 1e18) return 1e18;
            total += best;
        }
        return total;
    }

    // Wrapper to find shortest path and print XAI explanation
    void find_shortest_path_with_xai(const string& from, const string& to, int weightChoice) const {
        int s = find_node(from), t = find_node(to);
        if (s==-1 || t==-1) {
            cout << "Source or destination not found.\n";
            return;
        }
        // Run Dijkstra
        auto res = dijkstra(s, weightChoice);
        auto dist = res.first;
        auto prev = res.second;
        if (dist[t] >= 1e17) {
            cout << "No path found between " << from << " and " << to << ".\n";
            return;
        }
        auto path = reconstruct_path(s,t,prev);
        cout << "\n--- XAI Explanation (Dijkstra) ---\n";
        cout << "Metric used: " << (weightChoice==0?"Distance (km)":(weightChoice==1?"Travel time (min)":"Cost")) << "\n";
        cout << "Algorithm: Dijkstra's algorithm (priority queue-based).\n";
        // XAI: We describe why nodes were chosen/updated in algorithm comments in dijkstra()
        int reachable = 0;
        for (double d : dist) if (d < 1e17) reachable++;
        cout << "Nodes reachable during run: " << reachable << "\n";
        cout << "Total " << (weightChoice==0?"distance: ": (weightChoice==1?"travel time: ":"cost: ")) << dist[t];
        if (weightChoice==0) cout << " km\n"; else if (weightChoice==1) cout << " min\n"; else cout << " units\n";
        cout << "Chosen path: ";
        for (size_t i=0;i<path.size();++i) {
            cout << nodes[path[i]].name;
            if (i+1<path.size()) cout << " -> ";
        }
        cout << "\n";

        // Compare with BFS hop-count path
        auto bfs_p = bfs_path(s,t);
        if (!bfs_p.empty() && bfs_p != path) {
            double alt_metric = compute_metric_for_path(bfs_p, weightChoice);
            cout << "Alternative (BFS hop-count) path has " << bfs_p.size()-1 << " hops and metric value " << alt_metric << ".\n";
            cout << "XAI: Dijkstra path chosen because it has lower " << (weightChoice==0?"distance":(weightChoice==1?"travel time":"cost"))
                 << " than the BFS alternative by " << (alt_metric - dist[t]) << ".\n";
        } else {
            cout << "No alternative (BFS) path differs from Dijkstra path.\n";
        }
        cout << "-----------------------------------\n";
    }

    // Simple rule-based congestion predictor (explainable)
    double predict_congestion_probability(const Edge &e, int hour_of_day) const {
        // Base probability
        double p = 0.05;
        // Peak hour boost
        if ((hour_of_day >= 7 && hour_of_day <= 9) || (hour_of_day >= 16 && hour_of_day <= 19)) p += 0.4;
        // Speed heuristic (km/h) = distance_km / (travel_time/60)
        double speed = (e.travel_time > 0 ? e.distance_km / (e.travel_time / 60.0) : 0.0);
        if (speed < 20.0) p += 0.25; // slow roads -> likely congested
        else if (speed < 40.0) p += 0.1;
        if (p > 0.99) p = 0.99;
        // XAI: Predict using human-understandable rules: peak hours + low average speed.
        return p;
    }

    // Show top-k predicted congested routes
    void show_predicted_congestion(int hour_of_day, int topK = 5) const {
        vector<pair<double, pair<int, Edge>>> scored;
        for (int u=0; u<(int)adj.size(); ++u) {
            for (const auto &e : adj[u]) {
                double p = predict_congestion_probability(e, hour_of_day);
                scored.push_back({p,{u,e}});
            }
        }
        sort(scored.begin(), scored.end(), [](const auto &a, const auto &b){ return a.first > b.first;});
        cout << "\n--- Predicted Congestion (hour " << hour_of_day << ":00) ---\n";
        int show = min((int)scored.size(), topK);
        for (int i=0;i<show;++i) {
            const auto &it = scored[i];
            const Edge &e = it.second.second;
            int u = it.second.first;
            cout << "[" << e.id << "] " << nodes[u].name << " -> " << nodes[e.to].name << " (" << e.name << ") : "
                 << it.first*100 << "%\n";
            // XAI: explain reason
            cout << "  XAI: Reason: ";
            if ((hour_of_day >= 7 && hour_of_day <= 9) || (hour_of_day >= 16 && hour_of_day <= 19)) cout << "Peak hour; ";
            double speed = (e.travel_time > 0 ? e.distance_km / (e.travel_time / 60.0) : 0.0);
            if (speed < 20.0) cout << "low average speed (<20 km/h) indicates likely congestion.";
            else if (speed < 40.0) cout << "moderate speed (20-40 km/h) indicates possible slowing.";
            else cout << "speed appears good; congestion mostly due to timing.";
            cout << "\n";
        }
        cout << "--------------------------------------------\n";
    }

    // ASCII Visualization (places nodes on a small grid and draws straight lines)
    void ascii_visualize(int width = 41, int height = 21) {
        if (nodes.empty()) {
            cout << "No nodes to visualize.\n";
            return;
        }
        // Assign coordinates if not present (deterministic placement by hashing name)
        for (int i = 0; i < (int)nodes.size(); ++i) {
            if (nodes[i].x == -1 || nodes[i].y == -1) {
                // simple hash to place nodes across grid
                unsigned long h = 1469598103934665603ull;
                for (char c : nodes[i].name) h = (h ^ c) * 1099511628211ull;
                nodes[i].x = (h % (width-4)) + 2;
                nodes[i].y = ((h/13) % (height-4)) + 2;
            }
        }
        // create blank canvas
        vector<string> canvas(height, string(width, ' '));
        // draw edges as straight horizontal/vertical lines (approx)
        for (int u = 0; u < (int)nodes.size(); ++u) {
            for (const auto &e : adj[u]) {
                int v = e.to;
                int x1 = nodes[u].x, y1 = nodes[u].y;
                int x2 = nodes[v].x, y2 = nodes[v].y;
                // Draw simple line: move horizontally then vertically
                int x = x1, y = y1;
                while (x != x2) {
                    canvas[y][x] = (canvas[y][x] == ' ' ? '-' : canvas[y][x]);
                    x += (x2 > x ? 1 : -1);
                }
                while (y != y2) {
                    canvas[y][x] = (canvas[y][x] == ' ' ? '|' : canvas[y][x]);
                    y += (y2 > y ? 1 : -1);
                }
                // Put a connector at target
                canvas[y2][x2] = 'o';
            }
        }
        // place node labels (first character or up to 3 chars)
        for (int i = 0; i < (int)nodes.size(); ++i) {
            int x = nodes[i].x, y = nodes[i].y;
            string label = nodes[i].name;
            // keep up to 3 chars
            if (label.size() > 3) label = label.substr(0,3);
            for (size_t k = 0; k < label.size() && x + (int)k < width; ++k) {
                canvas[y][x+k] = label[k];
            }
        }
        // Print canvas
        cout << "\n---- ASCII Map (approx) ----\n";
        for (int r = 0; r < height; ++r) {
            cout << canvas[r] << "\n";
        }
        cout << "Legend: node labels (first 1-3 chars). 'o' marks edge endpoints where overlap happened.\n";
        cout << "Note: Visualization is approximate for small networks; used for quick inspection.\n";
        cout << "----------------------------\n";
    }

    // Seed sample network with coordinates to produce nicer ASCII art
    void seed_sample_network_with_coords() {
        // Clear existing
        nodes.clear(); adj.clear(); name_to_id.clear(); route_names.clear();
        next_edge_id = 1;
        // Create nodes with coordinates
        auto mk = [&](const string &name, int x, int y){
            int id = nodes.size();
            nodes.push_back({name,x,y});
            adj.emplace_back();
            name_to_id[name] = id;
            return id;
        };
        mk("A",4,4);
        mk("B",18,4);
        mk("C",30,6);
        mk("D",22,12);
        mk("E",6,14);
        mk("F",34,14);
        // add edges
        add_route("A","B",2.0,5.0,1.0,"A-B-1");
        add_route("B","C",3.0,7.0,1.5,"B-C-1");
        add_route("A","C",6.0,20.0,2.5,"A-C-1");
        add_route("C","D",1.5,4.0,0.8,"C-D-1");
        add_route("B","D",4.0,10.0,1.8,"B-D-1");
        add_route("D","E",2.5,6.0,1.2,"D-E-1");
        add_route("E","A",2.2,6.0,1.0,"E-A-1");
        add_route("C","F",4.0,12.0,2.0,"C-F-1");
        add_route("D","F",3.0,9.0,1.7,"D-F-1");
    }
};

/* -------------------------
   Menu & main program
   ------------------------- */

void print_main_menu() {
    cout << "\n================ Smart City Route Management ================\n";
    cout << "Menu:\n";
    cout << "1. Add a route\n";
    cout << "2. Remove a route\n";
    cout << "3. Update a route\n";
    cout << "4. View all routes\n";
    cout << "5. Find the shortest path (Dijkstra) with XAI\n";
    cout << "6. Sort routes by metric (functor)\n";
    cout << "7. Predict congestion (AI heuristic)\n";
    cout << "8. Undo last action\n";
    cout << "9. Redo last undone action\n";
    cout << "10. Seed sample network (with coords) + show ASCII map\n";
    cout << "11. Show ASCII visualization of current network\n";
    cout << "12. Exit\n";
    cout << "Enter option: ";
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << std::unitbuf; // ensures immediate output

    Graph g;
    cout << "Smart City Route Management System\n";
    cout << "Student: Brian Kodibona (24871125)\n";

    while (true) {
        print_main_menu();
        int opt;
        if (!(cin >> opt)) {
            cout << "Invalid input; exiting.\n";
            break;
        }
        if (opt == 1) {
            string from, to, rname;
            double dist, tmin, cost;
            cout << "FROM node name: "; cin >> from;
            cout << "TO node name: "; cin >> to;
            cout << "Route name (unique): "; cin >> rname;
            cout << "Distance (km): "; cin >> dist;
            cout << "Travel time (minutes): "; cin >> tmin;
            cout << "Cost (currency units): "; cin >> cost;
            // XAI: adding route because city official requested route registration.
            g.add_route(from, to, dist, tmin, cost, rname);
        } else if (opt == 2) {
            string from, rname;
            cout << "Origin node name: "; cin >> from;
            cout << "Route name to remove: "; cin >> rname;
            // XAI: removing allows dynamic network updates reflecting maintenance or closures.
            if (!g.remove_route(from, rname)) cout << "Route not found.\n";
        } else if (opt == 3) {
            string from, rname;
            cout << "Origin node name: "; cin >> from;
            cout << "Route name to update: "; cin >> rname;
            double nd, nt, nc;
            cout << "New distance (km): "; cin >> nd;
            cout << "New travel time (min): "; cin >> nt;
            cout << "New cost: "; cin >> nc;
            // XAI: update used to reflect changed road conditions or re-measurements.
            if (!g.update_route(from, rname, nd, nt, nc)) cout << "Route not found.\n";
        } else if (opt == 4) {
            g.print_all_routes();
        } else if (opt == 5) {
            string from, to;
            int metric;
            cout << "From: "; cin >> from;
            cout << "To: "; cin >> to;
            cout << "Choose metric: 0=Distance(km), 1=TravelTime(min), 2=Cost : ";
            cin >> metric;
            // XAI: We make metric explicit to be transparent about optimization target.
            g.find_shortest_path_with_xai(from, to, metric);
        } else if (opt == 6) {
            int metric;
            cout << "Sort by: 0=Distance(km),1=TravelTime(min),2=Cost : ";
            cin >> metric;
            g.sort_and_print_routes(metric);
        } else if (opt == 7) {
            int hour;
            cout << "Enter hour of day (0-23) for prediction: "; cin >> hour;
            if (hour < 0 || hour > 23) cout << "Invalid hour\n";
            else g.show_predicted_congestion(hour, 6);
        } else if (opt == 8) {
            // XAI: Undo allows experimenting with changes without permanent loss.
            g.undo();
        } else if (opt == 9) {
            g.redo();
        } else if (opt == 10) {
            g.seed_sample_network_with_coords();
            cout << "Sample network seeded. Displaying ASCII map:\n";
            g.ascii_visualize();
            g.print_all_routes();
        } else if (opt == 11) {
            g.ascii_visualize();
        } else if (opt == 12) {
            cout << "Exiting. Goodbye.\n";
            break;
        } else {
            cout << "Invalid option.\n";
        }
    }

    cout << "\nPress ENTER to exit...\n";
    cin.ignore();
    cin.get();
    return 0;
}

/* -------------------------
   Documentation (END OF FILE)
   -------------------------

Problem analysis and chosen solution approach:

- Representation:
  - Intersections are modelled as nodes (Node).
  - Routes are modelled as directed edges (Edge) with attributes: distance_km, travel_time (minutes), cost, name and id.
  - Core graph: adjacency list -> vector<list<Edge>>. The list provides efficient insertion and deletion when updating routes.

- Data structures used (at least three):
  - vector<Node> and vector<list<Edge>> (graphs & nodes).
  - list<Edge> (per-node adjacency lists).
  - set<string> route_names (unique route registry).
  - unordered_map<string,int> name_to_id (fast string->id mapping).
  - stack<Action> undo_stack, redo_stack (undo/redo functionality).
  - priority_queue used inside Dijkstra for efficient node selection.
  - queue used for BFS comparisons.

- Algorithms:
  - Dijkstra's algorithm implemented for general shortest-path queries with selectable weight (distance/time/cost).
  - BFS for hop-count comparison (unweighted).
  - Sorting performed with STL sort and a custom functor demonstrating functor usage.

- XAI principles applied:
  - All decision points (Dijkstra selection/update, sorting comparator, congestion predictions) have printed explanations.
  - Inline comments marked `// XAI:` explain what the code is doing and why (briefly).
  - After computing shortest path the program outputs: metric used, nodes reachable, total cost and path, and numeric comparison with BFS alternative when available.

- AI Integration (explainable):
  - A small rule-based congestion predictor uses:
    - Peak-hour boosts (7-9, 16-19).
    - Low average speed computed as (distance_km / (travel_time/60)) in km/h.
  - This is intentionally transparent (explainable) and printed with reasons for each flagged route.

- Visualization:
  - ASCII visualization plotted on a small grid for quick inspection.
  - Nodes are assigned simple deterministic coordinates (or can be seeded) and edges drawn as horizontal/vertical strokes.
  - This provides a lightweight visual check for network structure (bonus +5%).

- Undo/Redo:
  - Structural edits (add/remove/update) are recorded and reversible using stacks. This supports safe experimentation.

- How the program meets the deliverables:
  - Single `.cpp` file (this file).
  - Menu-driven interface with options to add/remove/update/view/find paths/sort/predict/visualize.
  - Dijkstra implemented and invoked from menu; XAI printed.
  - At least three data structures used and documented.
  - Code contains compulsory `// XAI:` comments at explanation points.
  - Documentation included at the end of the file (this section).

Testing / Example usage:
1. Compile: `g++ -std=c++17 24871125_FP.cpp -o route_manager`
2. Run: `./route_manager`
3. Choose option 10 to seed the sample network and show ASCII map.
4. Use option 5 to find shortest path (e.g. from A to F, metric 1 = travel time).
5. Use option 7 with hour 8 to see predicted congested routes and explanations.
6. Add / update / remove routes and use Undo/Redo to verify history.

Notes / Possible extensions:
- Export DOT format for Graphviz to produce nicer visualizations.
- Replace rule-based predictor with a learned model and include SHAP/feature importances for XAI.
- Add A* with heuristic (e.g., Euclidean based on coordinates) and compare to Dijkstra (performance + explanation).

End of documentation.
*/
