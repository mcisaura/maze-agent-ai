from typing import List, Tuple
from maze_reader import (
    load_maze,
    load_hazards,
    find_start_goal,
    update_fire_in_hazards,
    get_teleport_pairs,
    can_move,
    Hazard,
    GRID,
)

# ── Action constants ────────────────────────────────────────────────────────
# These numbers match what the agent sends in its action list
ACTION_UP    = 0
ACTION_DOWN  = 1
ACTION_LEFT  = 2
ACTION_RIGHT = 3

# How each action changes (row, col)
DELTAS = {
    ACTION_UP:    (-1,  0),   # row decreases (move toward row 0)
    ACTION_DOWN:  ( 1,  0),   # row increases (move toward row 63)
    ACTION_LEFT:  ( 0, -1),   # col decreases
    ACTION_RIGHT: ( 0,  1),   # col increases
}

# maze_reader.can_move() uses string directions
DIRECTION_NAMES = {
    ACTION_UP:    "up",
    ACTION_DOWN:  "down",
    ACTION_LEFT:  "left",
    ACTION_RIGHT: "right",
}

# Confusion inverts these pairs
INVERT_ACTION = {
    ACTION_UP:    ACTION_DOWN,
    ACTION_DOWN:  ACTION_UP,
    ACTION_LEFT:  ACTION_RIGHT,
    ACTION_RIGHT: ACTION_LEFT,
}


# ── TurnResult ──────────────────────────────────────────────────────────────

class TurnResult:
    """
    Returned by env.step() after every turn.
    This is ALL the agent knows about what just happened.

    Fields
    ------
    wall_hits        : how many of the submitted actions hit a wall (0-5)
    current_position : agent's (row, col) after the turn
    is_dead          : True if agent stepped on fire this turn
    is_confused      : True if agent stepped on a confusion trap this turn
    is_goal_reached  : True if agent reached the goal this turn
    teleported       : True if agent stepped on a teleporter this turn
    actions_executed : how many actions ran before death/goal (1-5)
    """
    def __init__(self):
        self.wall_hits:        int             = 0
        self.current_position: Tuple[int, int] = (0, 0)
        self.is_dead:          bool            = False
        self.is_confused:      bool            = False
        self.is_goal_reached:  bool            = False
        self.teleported:       bool            = False
        self.actions_executed: int             = 0

    def __repr__(self):
        return (
            f"TurnResult("
            f"pos={self.current_position}, "
            f"dead={self.is_dead}, "
            f"goal={self.is_goal_reached}, "
            f"wall_hits={self.wall_hits}, "
            f"teleported={self.teleported}, "
            f"confused={self.is_confused}, "
            f"actions={self.actions_executed})"
        )


# ── MazeEnvironment ─────────────────────────────────────────────────────────

class MazeEnvironment:
    """
    The maze world. Your agent calls:
        pos    = env.reset()          ← start a new episode
        result = env.step(actions)    ← submit 1-5 actions, get TurnResult back
        stats  = env.get_episode_stats()

    The agent is completely blind — it only learns about the maze
    through the TurnResult it gets back after each step.
    """

    def __init__(self, maze_path: str = "MAZE_0.png",
                       hazard_path: str = "MAZE_1.png"):

        # ── Load walls from MAZE_0.png ───────────────────────────────────────
        print(f"Loading maze walls  : {maze_path}")
        self.image, self.h_walls, self.v_walls = load_maze(maze_path)

        # ── Load hazards from MAZE_1.png ─────────────────────────────────────
        print(f"Loading hazards     : {hazard_path}")
        self.base_hazards = load_hazards(hazard_path)

        # Working copy of hazards — fire positions change each episode
        self.hazards     = dict(self.base_hazards)
        self.fire_pivots = None   # auto-detected on first rotation call

        # ── Find start and goal from wall gaps ───────────────────────────────
        self.start, self.goal = find_start_goal(self.h_walls)
        print(f"Start               : {self.start}")
        print(f"Goal                : {self.goal}")

        # ── Build teleporter map ─────────────────────────────────────────────
        # get_teleport_pairs() returns {color: (cell_a, cell_b)}
        # We flatten this into {cell_a: cell_b, cell_b: cell_a} for O(1) lookup
        self.teleport_pairs = get_teleport_pairs(self.hazards)
        self.teleport_map   = self._build_teleport_map(self.hazards)
        print(f"Teleporter pairs    : {len(self.teleport_map)//2} pairs")

        # ── Count hazards ────────────────────────────────────────────────────
        from collections import Counter
        counts = Counter(self.hazards.values())
        print(f"Fire cells          : {counts.get(Hazard.FIRE, 0)}")
        print(f"Confusion traps     : {counts.get(Hazard.CONFUSION, 0)}")
        print(f"Teleporters (cells) : {counts.get(Hazard.TP_GREEN,0) + counts.get(Hazard.TP_YELLOW,0) + counts.get(Hazard.TP_PURPLE,0) + counts.get(Hazard.TP_RED,0)}")
        print()

        # ── Episode state (reset each episode) ──────────────────────────────
        self.agent_pos      = self.start
        self.is_confused    = False   # is confusion currently active?
        self.confused_turns = 0       # turns of confusion remaining
        self.turn_count     = 0
        self.death_count    = 0
        self.confused_count = 0
        self.cells_visited  = []
        self.goal_reached   = False
        self.episode_number = 0

    # ────────────────────────────────────────────────────────────────────────
    # Teleport map builder
    # ────────────────────────────────────────────────────────────────────────

    def _build_teleport_map(self, hazards: dict) -> dict:
        """
        Build a flat lookup: stepping on cell A → lands on cell B.

        The spec says teleporters are one-way (pads don't exist at destinations)
        but maze_reader groups them as pairs, so we map both directions.
        Each color has exactly 2 cells that are linked to each other.
        """
        teleport_map = {}
        pairs = get_teleport_pairs(hazards)

        for color, pair in pairs.items():
            if len(pair) == 2:
                cell_a, cell_b = pair
                teleport_map[cell_a] = cell_b
                teleport_map[cell_b] = cell_a

        return teleport_map

    # ────────────────────────────────────────────────────────────────────────
    # Episode reset
    # ────────────────────────────────────────────────────────────────────────

    def reset(self) -> Tuple[int, int]:
        self.episode_number += 1
        self.hazards, self.fire_pivots = update_fire_in_hazards(
            self.hazards,
            self.fire_pivots
        )

        self.teleport_map = self._build_teleport_map(self.hazards)

        # ── Reset agent state ────────────────────────────────────────────────
        self.agent_pos      = self.start
        self.is_confused    = False
        self.confused_turns = 0
        self.turn_count     = 0
        self.death_count    = 0
        self.confused_count = 0
        self.cells_visited  = [self.start]
        self.goal_reached   = False

        return self.start

    # ────────────────────────────────────────────────────────────────────────
    # Step — the main function the agent calls every turn
    # ────────────────────────────────────────────────────────────────────────

    def step(self, actions: List[int]) -> TurnResult:

        if not actions:
            raise ValueError("Must submit at least 1 action per turn.")
        if len(actions) > 5:
            raise ValueError("Cannot submit more than 5 actions per turn.")

        result = TurnResult()
        self.turn_count += 1

        # Process each action one at a time
        for i, action in enumerate(actions):

            # ── WAIT: agent burns this step, nothing happens ─────────────────
            # Used to end the turn early and advance fire rotation
            if action == 4:   # ACTION_WAIT
                result.actions_executed = i + 1
                continue   # skip to next action slot

            # ── CONFUSION: invert the action if confused ─────────────────────
            effective_action = INVERT_ACTION[action] if self.is_confused else action

            # ── WALLS: try to move ────────────────────────────────────────────
            direction = DIRECTION_NAMES[effective_action]
            row, col  = self.agent_pos

            if not can_move(row, col, direction, self.h_walls, self.v_walls):
                # Wall blocks movement — agent stays, wall_hits increments
                result.wall_hits += 1
                result.actions_executed = i + 1
                continue   # move on to next action in the list

            # Movement successful — update position
            dr, dc         = DELTAS[effective_action]
            self.agent_pos = (row + dr, col + dc)
            self.cells_visited.append(self.agent_pos)

            # ── Check what's in the new cell ──────────────────────────────────
            cell_hazard = self.hazards.get(self.agent_pos)

            # ── TELEPORTER ────────────────────────────────────────────────────
            # Spec: "stepping on teleport pad instantly moves agent"
            # The agent lands on the pad, then immediately teleports.
            # current_position in TurnResult shows the DESTINATION.
            if self.agent_pos in self.teleport_map:
                destination    = self.teleport_map[self.agent_pos]
                self.agent_pos = destination
                self.cells_visited.append(destination)
                result.teleported = True

                # After teleporting, check what's at the destination
                cell_hazard = self.hazards.get(self.agent_pos)

            # ── CONFUSION TRAP ────────────────────────────────────────────────
            # Acts as a TOGGLE SWITCH:
            #   Hit confusion cell while normal   → controls inverted (ON)
            #   Hit confusion cell while confused → controls restored  (OFF)
            if cell_hazard == Hazard.CONFUSION:
                self.is_confused = not self.is_confused   # toggle
                result.is_confused  = True
                self.confused_count += 1

            # ── FIRE (death) ──────────────────────────────────────────────────
            # Spec: "agent dies instantly upon entering pit cell"
            # Remaining actions in the list are ignored
            # Agent respawns at start NEXT turn (not immediately)
            if cell_hazard == Hazard.FIRE:
                result.is_dead          = True
                result.current_position = self.agent_pos   # show WHERE agent died
                result.actions_executed = i + 1
                self.death_count       += 1

                # Respawn at start — agent will be here next turn
                self.agent_pos = self.start

                return result   # stop processing remaining actions

            # ── GOAL ──────────────────────────────────────────────────────────
            # Spec: "episode ends immediately" when goal is reached
            if self.agent_pos == self.goal:
                result.is_goal_reached  = True
                result.current_position = self.agent_pos
                result.actions_executed = i + 1
                self.goal_reached       = True

                return result   # stop processing remaining actions

            result.actions_executed = i + 1

        # Confusion is a toggle — no countdown needed

        result.current_position = self.agent_pos
        return result

    # ────────────────────────────────────────────────────────────────────────
    # Episode statistics
    # ────────────────────────────────────────────────────────────────────────

    def get_episode_stats(self) -> dict:
        """
        Return stats for the current episode.
        Matches the format defined in the spec (Section 6.1).
        """
        return {
            "turns_taken":    self.turn_count,
            "deaths":         self.death_count,
            "confused":       self.confused_count,
            "cells_explored": len(set(self.cells_visited)),   # unique cells
            "path_length":    len(self.cells_visited),        # including revisits
            "goal_reached":   self.goal_reached,
        }

    # ────────────────────────────────────────────────────────────────────────
    # Debug helpers
    # ────────────────────────────────────────────────────────────────────────

    def print_hazard_at(self, row: int, col: int):
        """Print what hazard (if any) is at a given cell."""
        hz = self.hazards.get((row, col))
        print(f"Cell ({row},{col}): {hz.value if hz else 'empty'}")

    def print_fire_positions(self):
        """Print all current fire cell positions."""
        fire = [(r,c) for (r,c), hz in self.hazards.items() if hz == Hazard.FIRE]
        print(f"Fire cells this episode ({len(fire)}): {sorted(fire)}")