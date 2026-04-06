import json
import heapq
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from maze_reader import (
    GRID, get_teleport_pairs, load_maze, load_hazards,
    get_start, get_goal, can_move, get_hazard, if_alive,
    update_fire_in_hazards, Hazard,
)

CELL_PX   = 20
SAVE_PATH = "results/maze_knowledge.json"

DIRECTIONS = {"up": (-1,0), "right": (0,1), "down": (1,0), "left": (0,-1)}
OPPOSITE   = {"up":"down", "down":"up", "left":"right", "right":"left"}

# ---------------------------------------------------------------------------
# Knowledge: what exits does each cell have?
# { (row,col): ["up", "right", ...] }
# ---------------------------------------------------------------------------

def save_knowledge(cell_exits, move_log):
    """Save discovered exits + this run's move log. Appends to existing file."""
    path = Path(SAVE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data or start fresh
    if path.exists():
        with open(path) as f:
            data = json.load(f)
    else:
        data = {"runs": 0, "cell_exits": {}, "move_history": []}

    # Merge cell_exits (union — never lose knowledge)
    for cell, dirs in cell_exits.items():
        key = str(cell)
        existing = set(data["cell_exits"].get(key, []))
        data["cell_exits"][key] = list(existing | set(dirs))

    # Append this run's moves
    run_number = data["runs"] + 1
    data["runs"] = run_number
    data["move_history"].append({
        "run": run_number,
        "moves": move_log,   # [{"step":1,"from":[r,c],"to":[r,c],"dir":"right"}, ...]
    })

    with open(SAVE_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[Save] Run #{run_number} | {len(cell_exits)} cells | {len(move_log)} moves → {SAVE_PATH}")


def load_knowledge():
    """Load cell exits. Returns (cell_exits, is_new_maze, past_runs)."""
    path = Path(SAVE_PATH)

    if not path.exists():
        print("[Memory] No save file → NEW maze")
        return defaultdict(set), True, 0

    with open(path) as f:
        data = json.load(f)

    cell_exits = defaultdict(set)
    for k, dirs in data.get("cell_exits", {}).items():
        row, col = (int(x) for x in k.strip("()").split(","))
        cell_exits[(row, col)] = set(dirs)

    runs = data.get("runs", 0)
    print(f"[Memory] Loaded {len(cell_exits)} cells | past runs: {runs}")
    return cell_exits, False, runs


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

class World:
    def __init__(self, h_walls, v_walls, hazards):
        self.h_walls = h_walls
        self.v_walls = v_walls
        self.hazards = hazards

    def can_move(self, row, col, direction):
        return can_move(row, col, direction, self.h_walls, self.v_walls)

    def get_hazard(self, row, col):
        return get_hazard(row, col, self.hazards)

    def is_alive(self, row, col):
        return if_alive(row, col, self.hazards)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, world, start, goal, cell_exits):
        self.world      = world
        self.start      = start
        self.goal       = goal
        self.position   = start
        self.cell_exits = cell_exits
        self.wall_hits = 0
        self.turns = 0
        self.hazard_hits = 0
        self.move_log = []
        self.solution_path = [] 
        self.path         = [start]

        self.path    = [start]
        self.visited = set()
        self.steps   = 0
        self.deaths  = 0

    def take_turn(self, actions):
        moves = 0
        i = 0

        self.turns += 1

        while moves < 5 and i < len(actions):
            action = actions[i]
            i += 1

            prev = self.position
            self._try_move(action)

            if self.position != prev:
                moves += 1

            if self.position == self.start and prev != self.start:
                break

            if self.position == self.goal:
                return True

        if moves == 5:
            self.world.hazards = update_fire_in_hazards(self.world.hazards)
            moves = 0

        return False

    def _try_move(self, direction):
        row, col = self.position
        dr, dc   = DIRECTIONS[direction]
        next_cell = (row + dr, col + dc)

        # WALL
        if not self.world.can_move(row, col, direction):
            self.wall_hits += 1
            return None

        # MOVE
        self.position = next_cell
        self.path.append(next_cell)
        self.steps += 1

        if self.steps % 5 == 0:
            self.turns += 1
            self.world.hazards = update_fire_in_hazards(self.world.hazards)

        self.move_log.append({
            "step": self.steps,
            "from": [row, col],
            "to":   list(next_cell),
            "dir":  direction,
        })

        self.cell_exits[(row, col)].add(direction)
        self.cell_exits[next_cell].add(OPPOSITE[direction])

        self.cell_exits[(row, col)].add(direction)

        # FIRE
        if not self.world.is_alive(*next_cell):
            self.deaths += 1
            self.position = self.start
            self.path.append(self.start)
            return None

        # HAZARDS
        hazard = self.world.get_hazard(*next_cell)
        if hazard is not None:
            self.hazard_hits += 1

            if hazard in {Hazard.TP_GREEN, Hazard.TP_YELLOW, Hazard.TP_PURPLE}:
                pairs = get_teleport_pairs(self.world.hazards)
                pair  = pairs[hazard]
                if len(pair) == 2:
                    dest = pair[0] if pair[1] == next_cell else pair[1]
                    self.position = dest
                    self.path.append(dest)

        return self.position

    def _astar(self):
        """Find shortest path using known exits."""
        def h(c):
            return abs(c[0] - self.goal[0]) + abs(c[1] - self.goal[1])

        heap   = [(h(self.start), 0, self.start, [self.start])]
        best_g = {self.start: 0}

        while heap:
            _, g, cell, route = heapq.heappop(heap)
            if cell == self.goal:
                return route
            for direction in self.cell_exits.get(cell, []):
                dr, dc     = DIRECTIONS[direction]
                neighbour  = (cell[0] + dr, cell[1] + dc)
                ng         = g + 1
                if ng < best_g.get(neighbour, float("inf")):
                    best_g[neighbour] = ng
                    heapq.heappush(heap, (ng + h(neighbour), ng, neighbour, route + [neighbour]))

        return None
    
    def _direction_between(self, a, b):
        dr = b[0] - a[0]
        dc = b[1] - a[1]
        for name, (r, c) in DIRECTIONS.items():
            if (r, c) == (dr, dc):
                return name
        return None

    def solve(self):
        print(f"Start: {self.start}  Goal: {self.goal}")

        found = False
        route = self._astar()

        if route:
            print(f"Known route: {len(route)-1} steps — walking it...")
            for i in range(len(route) - 1):
                self.position = route[i]
                direction = self._direction_between(route[i], route[i+1])
                if direction:
                    self._try_move(direction)
                if self.position == self.goal:
                    found = True
                    break
            if found:
                self.solution_path = route  # ← clean A* path

        # Always keep exploring to learn more
        self.visited.clear()
        self.position = self.start
        dfs_found = self._dfs(self.start)
        found = found or dfs_found

        if not self.solution_path and found:
            # First run — extract goal path from DFS via A* now that we learned
            self.solution_path = self._astar() or []

        print("\n===== AGENT SUMMARY =====")
        print(f"Result        : {'SUCCESS' if found else 'FAIL'}")
        print(f"Solution steps: {len(self.solution_path)-1 if self.solution_path else 'N/A'}")
        print(f"Steps         : {self.steps}")
        print(f"Turns         : {self.turns}")
        print(f"Wall hits     : {self.wall_hits}")
        print(f"Deaths        : {self.deaths}")
        print(f"Hazard hits   : {self.hazard_hits}")
        print(f"Visited cells : {len(self.visited)}")

        return found

    def _dfs(self, cell):
        self.visited.add(cell)
        found = (cell == self.goal)

        for direction in ["up", "right", "down", "left"]:
            dr, dc = DIRECTIONS[direction]
            next_cell = (cell[0] + dr, cell[1] + dc)

            if next_cell in self.visited:
                continue

            self.position = cell
            moved = self._try_move(direction)

            if moved is None:
                self.position = cell  # ← reset even on wall/fire
                continue

            current = self.position
            if current not in self.visited:
                if self._dfs(current):
                    found = True

            self.position = cell  # backtrack

        return found


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_path(h_walls, v_walls, path, start, goal, run=1, optimized=False):
    out = f"results/path_run{run}_{'optimized' if optimized else 'blind'}.png"
    size = GRID * CELL_PX
    img  = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 10)
    except Exception:
        font = ImageFont.load_default()

    for row in range(GRID):
        for col in range(GRID):
            x0, y0 = col*CELL_PX, row*CELL_PX
            fill = (0,200,0) if (row,col)==start else (200,0,0) if (row,col)==goal else (240,240,240)
            draw.rectangle([x0, y0, x0+CELL_PX, y0+CELL_PX], fill=fill)

    for wi in range(GRID+1):
        for col in range(GRID):
            if h_walls[wi, col]:
                draw.rectangle([col*CELL_PX, wi*CELL_PX, (col+1)*CELL_PX, wi*CELL_PX+2], fill=(0,0,0))
    for row in range(GRID):
        for wi in range(GRID+1):
            if v_walls[row, wi]:
                draw.rectangle([wi*CELL_PX, row*CELL_PX, wi*CELL_PX+2, (row+1)*CELL_PX], fill=(0,0,0))

    def center(cell):
        r, c = cell
        return (c*CELL_PX + CELL_PX//2, r*CELL_PX + CELL_PX//2)

    color = (0, 100, 255) if optimized else (255, 0, 0)
    width = 4             if optimized else 2

    for i in range(len(path)-1):
        draw.line([center(path[i]), center(path[i+1])], fill=color, width=width)

    # step numbers only on blind run; just endpoints on optimized
    if not optimized:
        for i, (r, c) in enumerate(path):
            draw.text((c*CELL_PX+2, r*CELL_PX+2), str(i), fill=(0,0,0), font=font)
    else:
        for i, (r, c) in enumerate(path):
            if i == 0 or i == len(path)-1:
                draw.text((c*CELL_PX+2, r*CELL_PX+2), str(i), fill=(0,0,180), font=font)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"Saved → {out}  ({len(path)-1} steps)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    image, h_walls, v_walls = load_maze("MAZE_0.png")
    hazards = load_hazards("MAZE_1.png")
    start   = get_start(h_walls)
    goal    = get_goal(h_walls)

    world = World(h_walls, v_walls, hazards)
    cell_exits, is_new_maze, past_runs = load_knowledge()

    if is_new_maze:
        print("Starting fresh on a new maze!")
    else:
        print(f"Resuming maze (run #{past_runs + 1})")

    agent = Agent(world, start, goal, cell_exits)
    found = agent.solve()

    current_run = past_runs + 1  # ← actual run number

    if found:
        save_knowledge(agent.cell_exits, agent.move_log)

    render_path(h_walls, v_walls, agent.path, start, goal,
                run=current_run,
                optimized=(found and len(agent.path) < 500))