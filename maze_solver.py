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

CELL_PX       = 20
SAVE_PATH     = "results/maze_knowledge.json"
DANGER_WEIGHT  = 5
REVISIT_WEIGHT = 3

DIRECTIONS = {"up": (-1,0), "right": (0,1), "down": (1,0), "left": (0,-1)}
OPPOSITE   = {"up":"down", "down":"up", "left":"right", "right":"left"}


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_knowledge(cell_exits, move_log):
    """Save discovered exits (with visit/death counts) and move log."""
    path = Path(SAVE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with open(path) as f:
            data = json.load(f)
    else:
        data = {"runs": 0, "cell_exits": {}, "move_history": []}

    # Merge cell_exits — each direction is {"visits": x, "deaths": y}
    for cell, dirs in cell_exits.items():
        key = str(cell)
        if key not in data["cell_exits"]:
            data["cell_exits"][key] = {}

        for direction, stats in dirs.items():
            if direction not in data["cell_exits"][key]:
                data["cell_exits"][key][direction] = {"visits": 0, "deaths": 0}

            saved = data["cell_exits"][key][direction]
            saved["visits"] = max(saved["visits"], stats["visits"])
            saved["deaths"] = max(saved["deaths"], stats["deaths"])

    # Append this run's moves
    run_number = data["runs"] + 1
    data["runs"] = run_number
    data["move_history"].append({"run": run_number, "moves": move_log})

    with open(SAVE_PATH, "w") as f:
        json.dump(data, f, indent=2)



def load_knowledge():
    """Load cell exits. Returns (cell_exits, is_new_maze, past_runs)."""
    path = Path(SAVE_PATH)

    if not path.exists():
        return defaultdict(dict), True, 0

    with open(path) as f:
        data = json.load(f)

    # cell_exits: { (row,col): { "up": {"visits":x, "deaths":y}, ... } }
    cell_exits = defaultdict(dict)
    for k, dirs in data.get("cell_exits", {}).items():
        row, col = (int(x) for x in k.strip("()").split(","))
        if isinstance(dirs, list):
            dirs = {d: {"visits": 0, "deaths": 0} for d in dirs}
        cell_exits[(row, col)] = dirs

    runs = data.get("runs", 0)
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
        self.cell_exits = cell_exits   # { (row,col): { "up": {"visits":x,"deaths":y} } }

        self.wall_hits      = 0
        self.turns          = 0
        self.steps          = 0
        self.deaths         = 0
        self.fire_hits      = 0
        self.teleport_hits  = 0
        self.confusion_hits = 0

        self.move_log      = []
        self.path          = [start]
        self.visited_count = defaultdict(int)

    def _astar(self):
        """Find shortest path from start to goal using known exits."""
        def h(c):
            return abs(c[0] - self.goal[0]) + abs(c[1] - self.goal[1])

        heap   = [(h(self.start), 0, self.start, [self.start])]
        best_g = {self.start: 0}

        while heap:
            _, g, cell, route = heapq.heappop(heap)
            if cell == self.goal:
                return route
            for direction in self.cell_exits.get(cell, {}):
                dr, dc    = DIRECTIONS[direction]
                neighbour = (cell[0] + dr, cell[1] + dc)
                ng        = g + 1
                if ng < best_g.get(neighbour, float("inf")):
                    best_g[neighbour] = ng
                    heapq.heappush(heap, (ng + h(neighbour), ng, neighbour, route + [neighbour]))
        return None

    def _print_summary(self, result, runtime):
        print("\n========== RESULTS ==========")
        print(f"Result      : {'SUCCESS' if result else 'FAILED'}")
        print(f"Steps       : {self.steps}")
        print(f"Turns       : {self.turns}")
        print(f"Wall hits   : {self.wall_hits}")
        print(f"Deaths      : {self.deaths}")
        print(f"  Fire hits       : {self.fire_hits}")
        print(f"  Teleport hits   : {self.teleport_hits}")
        print(f"  Confusion hits  : {self.confusion_hits}")
        print(f"Runtime     : {runtime:.2f}s")
        print("=============================\n")

    def solve(self):
        """Run 1: blind exploration. Run 2+: A* on known exits, then explore."""
        import time
        MAX_TURNS  = 10_000
        start_time = time.time()

        # ── If we have prior knowledge, try A* first ────────────────────────
        if self.cell_exits:
            route = self._astar()
            if route:
                moves_this_turn = 0
                for i in range(len(route) - 1):
                    next_cell = route[i + 1]
                    self.position = next_cell
                    self.visited_count[next_cell] += 1
                    self.path.append(next_cell)
                    self.move_log.append(list(next_cell))
                    self.steps += 1
                    moves_this_turn += 1

                    if moves_this_turn == 5:
                        self.turns += 1
                        moves_this_turn = 0
                        self.world.hazards = update_fire_in_hazards(self.world.hazards)

                    if self.position == self.goal:
                        if moves_this_turn > 0:
                            self.turns += 1  # count the partial turn
                        self._print_summary(True, time.time() - start_time)
                        return True
                self.position = self.start

        # ── Blind / fallback exploration ─────────────────────────────────────
        for _ in range(MAX_TURNS):
            self.turns += 1

            moves_made = 0
            while moves_made < 5:
                moved = self.move()
                if moved:
                    moves_made += 1

                if self.position == self.start and moves_made > 0:
                    break

                if self.position == self.goal:
                    self._print_summary(True, time.time() - start_time)
                    return True

            self.world.hazards = update_fire_in_hazards(self.world.hazards)

        self._print_summary(False, time.time() - start_time)
        return False

    def explore_theCell(self):
        """Sense all 4 directions from current cell. Does NOT move the agent."""
        row, col = self.position

        if not self.world.is_alive(row, col):
            self.fire_hits += 1
            self.deaths += 1
            save_knowledge(self.cell_exits, self.move_log)
            self.position = self.start
            return

        for direction in DIRECTIONS:
            if self.world.can_move(row, col, direction):
                if direction not in self.cell_exits[(row, col)]:
                    self.cell_exits[(row, col)][direction] = {"visits": 0, "deaths": 0}
            else:
                self.wall_hits += 1

    def move(self):
        """Pick the best direction and move. Returns True if moved."""
        row, col = self.position
        cell_key = (row, col)          # ← tuple, NOT str((row, col))

        # ── 1. If we haven't explored this cell yet, explore first ──────────
        if cell_key not in self.cell_exits:
            self.explore_theCell()
            return False

        # ── 2. Get known exits for this cell ────────────────────────────────
        known_exits = self.cell_exits[cell_key]  # {"up": {"visits":x,"deaths":y}, ...}

        if not known_exits:
            return False

        # ── 3. Score each exit ───────────────────────────────────────────────
        def score(direction):
            dr, dc = DIRECTIONS[direction]
            next_row, next_col = row + dr, col + dc

            dist    = abs(next_row - self.goal[0]) + abs(next_col - self.goal[1])
            stats   = known_exits[direction]
            danger  = stats["deaths"] / (stats["visits"] + 1)
            revisit = self.visited_count[(next_row, next_col)]

            return dist + (danger * DANGER_WEIGHT) + (revisit * REVISIT_WEIGHT)

        best_direction = min(known_exits, key=score)

        # ── 4. Actually move ─────────────────────────────────────────────────
        dr, dc = DIRECTIONS[best_direction]
        new_row, new_col = row + dr, col + dc

        self.cell_exits[cell_key][best_direction]["visits"] += 1
        self.position = (new_row, new_col)
        self.visited_count[self.position] += 1
        self.path.append(self.position)
        self.move_log.append([new_row, new_col])
        self.steps += 1

        # ── 5. Check hazards after moving ───────────────────────────────────
        if not self.world.is_alive(new_row, new_col):
            self.cell_exits[cell_key][best_direction]["deaths"] += 1
            self.fire_hits += 1
            self.deaths += 1
            save_knowledge(self.cell_exits, self.move_log)
            self.position = self.start
            self.path.append(self.start)
            return False

        hazard = self.world.get_hazard(new_row, new_col)
        if hazard == Hazard.CONFUSION:
            self.confusion_hits += 1
        elif hazard in {Hazard.TP_GREEN, Hazard.TP_YELLOW, Hazard.TP_PURPLE}:
            self.teleport_hits += 1
            pairs = get_teleport_pairs(self.world.hazards)
            pair  = pairs.get(hazard, [])
            if len(pair) == 2:
                dest = pair[0] if pair[1] == self.position else pair[1]
                self.position = dest
                self.path.append(dest)
        return True


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

    if not optimized:
        for i, (r, c) in enumerate(path):
            draw.text((c*CELL_PX+2, r*CELL_PX+2), str(i), fill=(0,0,0), font=font)
    else:
        for i, (r, c) in enumerate(path):
            if i == 0 or i == len(path)-1:
                draw.text((c*CELL_PX+2, r*CELL_PX+2), str(i), fill=(0,0,180), font=font)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


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


    agent = Agent(world, start, goal, cell_exits)
    found = agent.solve()

    current_run = past_runs + 1
    save_knowledge(agent.cell_exits, agent.move_log)

    # ── Always render the path the agent actually took ───────────────────
    render_path(h_walls, v_walls, agent.path, start, goal,
                run=current_run,
                optimized=False)   # blind run = red path with step numbers

    # ── If it found the goal, also render a clean version ───────────────
    if found:
        render_path(h_walls, v_walls, agent.path, start, goal,
                    run=current_run,
                    optimized=True)  # blue path, just endpoints labeled