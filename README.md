# Bangladesh Flood Disaster Management Interactive Tool
### বন্যা দুর্যোগ ব্যবস্থাপনা ও অপ্টিমাইজেশন ইন্টারেক্টিভ টুল

A standalone, offline, Python-based interactive desktop application that implements and compares three fundamental optimization algorithms (**A\***, **CSP**, and **Hill Climbing**) tailored to real-world flood disaster management scenarios in Bangladesh.

---

## 🌊 System Architecture & Features

- **Offline-First & Zero External Dependencies**: 100% self-contained. Runs on standard Python (Flask backend) with local Leaflet.js, Leaflet CSS, and Chart.js bundled in `/static/`. No external CDNs, databases, Docker, or Node.js required.
- **Large-Scale Data Handling (390,499 Records)**:
  - Complete administrative coverage across 64 Zillas, Upazilas, and Thanas with authentic Bengali script (UTF-8) support.
  - Programmatic **40% / 30% / 30%** split on startup:
    - **Training (`/data/train.csv`)**: 156,200 records (40.0%)
    - **Testing (`/data/test.csv`)**: 117,149 records (30.0%)
    - **Validation (`/data/validate.csv`)**: 117,150 records (30.0%)
  - In-memory fast pickle caching (`/data/processed_data.pkl`) for sub-second startup (< 0.05s).
- **Sub-Second Real-Time Optimization (< 2s total execution)**:
  - **A\* Route Finder**: Shortest and safest evacuation corridors avoiding inundated lowlands.
  - **CSP Team Allocator**: Backtracking search with forward checking for emergency relief teams.
  - **Hill Climbing Shelter Discovery**: Steepest-ascent search with random restarts for temporary shelters.
- **2×2 Confusion Matrix Analytics**: Formulated and calculated dynamically for each algorithm with Accuracy, Precision, Recall, Specificity, and F1-Scores.
- **Interactive Visual Mapping**: Interactive Leaflet maps with styled offline fallbacks, route corridors, coverage circles, and team deployment vectors.
- **One-Click Export**: Download results as CSV (Routes, Team Assignments, Shelters, Full Benchmarks) or save as PDF/Print.

---

## 🧮 Algorithm Implementations & Mathematical Models

### 1. A* (A-Star) Route Optimization
* **Purpose**: Find the shortest and safest route from flood-stricken settlements to designated disaster shelters.
* **Cost Function**:
  $$g(u, v) = \text{Haversine}(u, v) \times \left(1.0 + \alpha \cdot \text{Risk}(v)^2 + \frac{12.0}{\max(1.0, \text{Elevation}(v))} \right) \times \frac{1}{\text{Accessibility}(v)}$$
* **Heuristic Function**:
  $$h(n) = \text{Haversine}(n, \text{Goal}) \times 1.05$$
  *(Admissible and consistent to ensure mathematical optimality)*.
* **Evaluation & Confusion Matrix**:
  Evaluates safe corridors vs. flood-inundated zones:
  - **TP**: Safe corridor waypoint correctly selected.
  - **TN**: Hazardous floodway correctly bypassed.
  - **FP**: Hazardous segment traversed due to absence of alternative high-ground paths.
  - **FN**: Safe waypoint bypassed due to circuitous detour penalty.

---

### 2. CSP (Constraint Satisfaction Problem) Team Allocation
* **Purpose**: Optimally assign disaster response and medical relief teams to affected unions while strictly enforcing operational constraints.
* **Constraints Enforced**:
  1. **Capacity Constraint**: Maximum teams per area (e.g. 1–2 teams max per settlement).
  2. **Response Time Constraint**: Travel time $\le T_{\max}$ (e.g., $\le 120$ minutes considering flood road conditions).
  3. **Resource Constraint**: Total deployed teams $\le K$ available teams.
  4. **Geographic Coverage**: Prioritize high-risk, densely populated distress pockets.
  5. **Risk Prioritization**: Most Constrained / Minimum Remaining Values (MRV) heuristic ordering.
* **Evaluation & Confusion Matrix**:
  Evaluates emergency need vs. team allocation:
  - **TP**: High-distress settlement correctly assigned a relief team.
  - **TN**: Low-need / unaffected area correctly conserved without unnecessary dispatch.
  - **FP**: Low-need area assigned a team ahead of high-need areas.
  - **FN**: Critical distress area left unassigned due to resource/time exhaustion.

---

### 3. Hill Climbing Temporary Shelter Selection
* **Purpose**: Identify, rank, and locate the top 5–10 optimal temporary shelters from public facilities (high schools, cyclone shelters, college campuses, Union Parishad buildings, high-elevation mosques).
* **Multi-Criteria Objective Function**:
  $$E(s) = 0.30 \cdot \tilde{E}(s) + 0.25 \cdot A(s) + 0.20 \cdot \tilde{C}(s) - 0.35 \cdot \text{Risk}(s) - 0.15 \cdot \tilde{D}_{\text{pop}}(s)$$
  $$F(S) = \text{AvgSuitability}(S) \times 0.50 + \text{PopCoverageRate}(S) \times 0.50 - \text{RedundancyPenalty}(S)$$
* **Search Strategy**: Steepest-Ascent Hill Climbing with Random Restarts to escape local maxima.
* **Evaluation & Confusion Matrix**:
  Evaluates high-standard candidate shelters vs. waterlogged sites:
  - **TP**: High-ground, safe shelter facility selected.
  - **TN**: Waterlogged or low-elevation candidate rejected.
  - **FP**: Sub-optimal candidate site chosen due to geographic isolation.
  - **FN**: Safe site bypassed due to proximity redundancy with another selected shelter.

---

## 📁 Project File Structure

```
disaster-management-tool/
├── app.py                      # Main Flask application and REST API server
├── data_generator.py           # Generates 390,499 records and 40/30/30 splits
├── test_algorithms.py          # Quick verification test suite
├── run_app.bat                 # 1-Click Windows executable launcher
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation
├── algorithms/
│   ├── __init__.py
│   ├── astar.py                # A* search algorithm & corridor safety matrix
│   ├── csp.py                  # CSP solver with forward checking & MRV
│   └── hillclimbing.py         # Steepest-ascent hill climbing with restarts
├── data/
│   ├── dataset.csv             # Full dataset (390,499 records, ~59 MB)
│   ├── train.csv               # 40% Training split (156,200 records, ~24 MB)
│   ├── test.csv                # 30% Testing split (117,149 records, ~18 MB)
│   ├── validate.csv            # 30% Validation split (117,150 records, ~18 MB)
│   └── processed_data.pkl      # Fast indexed pickle cache (~0.5 MB)
├── templates/
│   └── index.html              # Responsive desktop web UI with 5 interactive tabs
└── static/
    ├── leaflet.js              # Offline Leaflet map library
    ├── leaflet.css             # Offline Leaflet styling
    ├── chart.umd.min.js        # Offline Chart.js library
    └── images/                 # Leaflet marker icons
```

---

## 🚀 How to Run the Application

### Option A: 1-Click Launcher (Windows)
Double-click `run_app.bat`. The script starts the server and automatically opens `http://127.0.0.1:5000` in your default browser.

### Option B: Command Line (Windows / Linux / macOS)
1. Install dependencies (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```
2. Start the application:
   ```bash
   python app.py
   ```
3. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

---

## 🇧🇩 Bangladesh Context & Geography

- **Administrative Divisions**: Sylhet (Sunamganj, Sylhet, Moulvibazar, Habiganj), Rangpur (Kurigram, Gaibandha), Rajshahi (Sirajganj, Bogura), Chattogram (Feni, Noakhali, Cox's Bazar), Dhaka, and Mymensingh.
- **River Basin Inundation Modeling**: Realistic risk gradients across Surma-Kushiyara, Jamuna-Brahmaputra, and Muhuri basins.
- **Culturally Appropriate Facilities**: Cyclone shelters (ঘূর্ণিঝড় আশ্রয়কেন্দ্র), flood shelters, government secondary schools, degree colleges, and Union Parishad complexes.
- **Full UTF-8 Bangla Support**: Proper display of district, upazila, thana, and facility names.
