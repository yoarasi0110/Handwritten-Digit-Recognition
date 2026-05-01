#!/usr/bin/env python3
"""3x3 Rubik's Cube scrambler + solver animation.

Solver strategy used here:
1. Generate a random scramble sequence.
2. Apply scramble to a solved cube.
3. Solve by applying the inverse sequence (guaranteed solution for generated scrambles).
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

Vec3 = Tuple[int, int, int]


@dataclass
class Sticker:
    pos: Vec3
    normal: Vec3
    color: str


BASE_MOVES = ["U", "D", "L", "R", "F", "B"]
MOVE_SUFFIXES = ["", "'", "2"]

# Singmaster face colors
FACE_COLORS: Dict[str, str] = {
    "U": "#ffffff",  # white
    "D": "#ffd500",  # yellow
    "F": "#009b48",  # green
    "B": "#0046ad",  # blue
    "L": "#ff5800",  # orange
    "R": "#b71234",  # red
}

# Net layout offsets (x, y) in sticker units
NET_OFFSETS = {
    "U": (3, 0),
    "L": (0, 3),
    "F": (3, 3),
    "R": (6, 3),
    "B": (9, 3),
    "D": (3, 6),
}


class RubiksCube:
    def __init__(self) -> None:
        self.stickers: List[Sticker] = []
        self._init_solved_state()

    def _init_solved_state(self) -> None:
        self.stickers.clear()
        for x in (-1, 0, 1):
            for z in (-1, 0, 1):
                self.stickers.append(Sticker((x, 1, z), (0, 1, 0), FACE_COLORS["U"]))
                self.stickers.append(Sticker((x, -1, z), (0, -1, 0), FACE_COLORS["D"]))
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                self.stickers.append(Sticker((x, y, 1), (0, 0, 1), FACE_COLORS["F"]))
                self.stickers.append(Sticker((x, y, -1), (0, 0, -1), FACE_COLORS["B"]))
        for z in (-1, 0, 1):
            for y in (-1, 0, 1):
                self.stickers.append(Sticker((-1, y, z), (-1, 0, 0), FACE_COLORS["L"]))
                self.stickers.append(Sticker((1, y, z), (1, 0, 0), FACE_COLORS["R"]))

    @staticmethod
    def _rot_plus_90(v: Vec3, axis: str) -> Vec3:
        x, y, z = v
        if axis == "x":
            return (x, -z, y)
        if axis == "y":
            return (z, y, -x)
        if axis == "z":
            return (-y, x, z)
        raise ValueError(f"Unknown axis: {axis}")

    @classmethod
    def _rotate_vec(cls, v: Vec3, axis: str, quarter_turns: int) -> Vec3:
        quarter_turns %= 4
        result = v
        for _ in range(quarter_turns):
            result = cls._rot_plus_90(result, axis)
        return result

    @staticmethod
    def _move_to_rotation(move: str) -> Tuple[str, int, int, int]:
        face = move[0]
        suffix = move[1:] if len(move) > 1 else ""

        base_map = {
            "U": ("y", 1, -1),
            "D": ("y", -1, 1),
            "R": ("x", 1, -1),
            "L": ("x", -1, 1),
            "F": ("z", 1, -1),
            "B": ("z", -1, 1),
        }
        if face not in base_map:
            raise ValueError(f"Unsupported move: {move}")

        axis, layer, sign = base_map[face]
        if suffix == "'":
            sign *= -1
            turns = 1
        elif suffix == "2":
            turns = 2
        elif suffix == "":
            turns = 1
        else:
            raise ValueError(f"Unsupported move suffix in: {move}")

        return axis, layer, sign, turns

    def apply_move(self, move: str) -> None:
        axis, layer, sign, turns = self._move_to_rotation(move)
        quarter_turns = (sign * turns) % 4

        for sticker in self.stickers:
            coord = {"x": sticker.pos[0], "y": sticker.pos[1], "z": sticker.pos[2]}[axis]
            if coord == layer:
                sticker.pos = self._rotate_vec(sticker.pos, axis, quarter_turns)
                sticker.normal = self._rotate_vec(sticker.normal, axis, quarter_turns)

    def apply_moves(self, moves: Iterable[str]) -> None:
        for mv in moves:
            self.apply_move(mv)

    def copy(self) -> "RubiksCube":
        clone = RubiksCube()
        clone.stickers = [Sticker(s.pos, s.normal, s.color) for s in self.stickers]
        return clone

    @staticmethod
    def inverse_move(move: str) -> str:
        if move.endswith("2"):
            return move
        if move.endswith("'"):
            return move[0]
        return move + "'"

    @classmethod
    def inverse_sequence(cls, moves: Sequence[str]) -> List[str]:
        return [cls.inverse_move(m) for m in reversed(moves)]

    def _face_row_col(self, sticker: Sticker) -> Tuple[str, int, int]:
        x, y, z = sticker.pos
        nx, ny, nz = sticker.normal

        if (nx, ny, nz) == (0, 1, 0):  # U
            return "U", z + 1, x + 1
        if (nx, ny, nz) == (0, -1, 0):  # D
            return "D", 1 - z, x + 1
        if (nx, ny, nz) == (0, 0, 1):  # F
            return "F", 1 - y, x + 1
        if (nx, ny, nz) == (0, 0, -1):  # B
            return "B", 1 - y, 1 - x
        if (nx, ny, nz) == (-1, 0, 0):  # L
            return "L", 1 - y, z + 1
        if (nx, ny, nz) == (1, 0, 0):  # R
            return "R", 1 - y, 1 - z

        raise ValueError("Invalid sticker normal")

    def facelet_map(self) -> Dict[Tuple[str, int, int], str]:
        result: Dict[Tuple[str, int, int], str] = {}
        for s in self.stickers:
            face, row, col = self._face_row_col(s)
            result[(face, row, col)] = s.color
        return result


def random_scramble(length: int, seed: int | None = None) -> List[str]:
    rng = random.Random(seed)
    scramble: List[str] = []

    prev_face = None
    for _ in range(length):
        face_choices = [f for f in BASE_MOVES if f != prev_face]
        face = rng.choice(face_choices)
        suffix = rng.choice(MOVE_SUFFIXES)
        scramble.append(face + suffix)
        prev_face = face

    return scramble


def draw_cube_net(ax, cube: RubiksCube) -> None:
    from matplotlib.patches import Rectangle

    ax.clear()
    ax.set_aspect("equal")
    ax.axis("off")

    facelets = cube.facelet_map()
    size = 1.0

    for face, (ox, oy) in NET_OFFSETS.items():
        for row in range(3):
            for col in range(3):
                x = ox + col * size
                y = oy + row * size
                color = facelets[(face, row, col)]
                rect = Rectangle((x, y), size, size, facecolor=color, edgecolor="black", linewidth=1.2)
                ax.add_patch(rect)

    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(9.5, -0.5)


def _vec_add(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_scale(v: Tuple[float, float, float], s: float) -> Tuple[float, float, float]:
    return (v[0] * s, v[1] * s, v[2] * s)


def draw_cube_3d(ax, cube: RubiksCube, azim: float) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ax.clear()
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_zlim(-1.8, 1.8)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=24, azim=azim)

    for sticker in cube.stickers:
        x, y, z = sticker.pos
        nx, ny, nz = sticker.normal

        center = (x + 0.52 * nx, y + 0.52 * ny, z + 0.52 * nz)

        if abs(nx) == 1:
            u, v = (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        elif abs(ny) == 1:
            u, v = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
        else:
            u, v = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)

        hs = 0.44
        corners = [
            _vec_add(_vec_add(center, _vec_scale(u, hs)), _vec_scale(v, hs)),
            _vec_add(_vec_add(center, _vec_scale(u, hs)), _vec_scale(v, -hs)),
            _vec_add(_vec_add(center, _vec_scale(u, -hs)), _vec_scale(v, -hs)),
            _vec_add(_vec_add(center, _vec_scale(u, -hs)), _vec_scale(v, hs)),
        ]

        poly = Poly3DCollection([corners], facecolors=sticker.color, edgecolors="black", linewidths=0.8)
        ax.add_collection3d(poly)


def _rotate_vec_float(v: Tuple[float, float, float], axis: str, angle_deg: float) -> Tuple[float, float, float]:
    x, y, z = v
    a = math.radians(angle_deg)
    c = math.cos(a)
    s = math.sin(a)
    if axis == "x":
        return (x, y * c - z * s, y * s + z * c)
    if axis == "y":
        return (x * c + z * s, y, -x * s + z * c)
    if axis == "z":
        return (x * c - y * s, x * s + y * c, z)
    raise ValueError(f"Unknown axis: {axis}")


def draw_cube_3d_moving(
    ax,
    cube: RubiksCube,
    active_rotation: Tuple[str, int, float] | None,
    azim: float,
) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ax.clear()
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_zlim(-1.8, 1.8)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=24, azim=azim)

    rot_axis = None
    rot_layer = None
    rot_angle = 0.0
    if active_rotation is not None:
        rot_axis, rot_layer, rot_angle = active_rotation

    for sticker in cube.stickers:
        px, py, pz = sticker.pos
        nx, ny, nz = sticker.normal
        pos = (float(px), float(py), float(pz))
        normal = (float(nx), float(ny), float(nz))

        if rot_axis is not None:
            coord = {"x": px, "y": py, "z": pz}[rot_axis]
            if coord == rot_layer:
                pos = _rotate_vec_float(pos, rot_axis, rot_angle)
                normal = _rotate_vec_float(normal, rot_axis, rot_angle)

        center = (
            pos[0] + 0.52 * normal[0],
            pos[1] + 0.52 * normal[1],
            pos[2] + 0.52 * normal[2],
        )

        if abs(normal[0]) > 0.5:
            u, v = (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        elif abs(normal[1]) > 0.5:
            u, v = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
        else:
            u, v = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)

        hs = 0.44
        corners = [
            _vec_add(_vec_add(center, _vec_scale(u, hs)), _vec_scale(v, hs)),
            _vec_add(_vec_add(center, _vec_scale(u, hs)), _vec_scale(v, -hs)),
            _vec_add(_vec_add(center, _vec_scale(u, -hs)), _vec_scale(v, -hs)),
            _vec_add(_vec_add(center, _vec_scale(u, -hs)), _vec_scale(v, hs)),
        ]

        poly = Poly3DCollection([corners], facecolors=sticker.color, edgecolors="black", linewidths=0.8)
        ax.add_collection3d(poly)


def animate_solution(
    scramble_moves: Sequence[str],
    solve_moves: Sequence[str],
    interval_ms: int,
    show: bool,
    save_path: str | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from matplotlib.widgets import Button, Slider
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script requires matplotlib (and pillow when using --save). "
            "Please install with: pip install matplotlib pillow"
        ) from exc

    cube = RubiksCube()
    cube.apply_moves(scramble_moves)

    frames: List[Dict[str, object]] = []
    frames_per_move = 8

    frames.append(
        {
            "net_cube": cube.copy(),
            "render_cube": cube.copy(),
            "active_rotation": None,
            "title": "Scrambled cube",
        }
    )

    for idx, mv in enumerate(solve_moves, start=1):
        base = cube.copy()
        axis, layer, sign, turns = RubiksCube._move_to_rotation(mv)
        target_angle = 90 * sign * turns
        for sub in range(1, frames_per_move + 1):
            progress = sub / frames_per_move
            if sub == frames_per_move:
                cube.apply_move(mv)
                net_cube = cube.copy()
            else:
                net_cube = base.copy()
            frames.append(
                {
                    "net_cube": net_cube,
                    "render_cube": base.copy(),
                    "active_rotation": (axis, layer, target_angle * progress),
                    "title": f"Solving step {idx}/{len(solve_moves)}: {mv}",
                }
            )

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[18, 1], hspace=0.15)
    ax_net = fig.add_subplot(gs[0, 0])
    ax_3d = fig.add_subplot(gs[0, 1], projection="3d")

    ax_start_btn = fig.add_axes([0.12, 0.03, 0.12, 0.05])
    ax_replay_btn = fig.add_axes([0.26, 0.03, 0.12, 0.05])
    ax_speed_slider = fig.add_axes([0.48, 0.04, 0.4, 0.03])

    start_btn = Button(ax_start_btn, "Start")
    replay_btn = Button(ax_replay_btn, "Replay")
    speed_slider = Slider(ax_speed_slider, "Speed", 0.25, 3.0, valinit=1.0, valstep=0.05)

    state = {
        "ended": False,
        "started": False,
        "frame": 0,
    }

    def update(frame_idx: int) -> None:
        state["frame"] = frame_idx
        frame_data = frames[frame_idx]
        draw_cube_net(ax_net, frame_data["net_cube"])
        draw_cube_3d_moving(
            ax_3d,
            frame_data["render_cube"],
            frame_data["active_rotation"],
            azim=42,
        )
        title = str(frame_data["title"])
        ax_net.set_title(f"2D Net - {title}", fontsize=12)
        ax_3d.set_title("3D Real Move Rotation", fontsize=12)
        if frame_idx == len(frames) - 1:
            state["ended"] = True
            replay_btn.label.set_text("Replay ✓")
        else:
            replay_btn.label.set_text("Replay")

    # Keep first frame visible, wait for user to click Start.
    update(0)

    timer = fig.canvas.new_timer(interval=interval_ms)

    def on_timer() -> None:
        if not state["started"] or state["ended"]:
            return
        if state["frame"] < len(frames) - 1:
            state["frame"] += 1
            update(state["frame"])
            fig.canvas.draw_idle()
        if state["frame"] >= len(frames) - 1:
            state["ended"] = True
            replay_btn.label.set_text("Replay ✓")
            fig.canvas.draw_idle()

    timer.add_callback(on_timer)
    timer.start()

    def on_start(_event) -> None:
        if state["started"] and not state["ended"] and state["frame"] > 0:
            return
        if state["ended"]:
            state["ended"] = False
            state["frame"] = 0
            update(0)
        replay_btn.label.set_text("Replay")
        state["started"] = True
        fig.canvas.draw_idle()

    def on_replay(_event) -> None:
        if not state["ended"]:
            return
        state["ended"] = False
        state["started"] = True
        state["frame"] = 0
        update(0)
        replay_btn.label.set_text("Replay")
        fig.canvas.draw_idle()

    def on_speed_change(multiplier: float) -> None:
        new_interval = max(1, int(interval_ms / max(0.05, multiplier)))
        timer.interval = new_interval

    start_btn.on_clicked(on_start)
    replay_btn.on_clicked(on_replay)
    speed_slider.on_changed(on_speed_change)

    # Hold references so widgets callbacks are not garbage-collected.
    fig._cube_controls = (start_btn, replay_btn, speed_slider, timer)  # type: ignore[attr-defined]

    if save_path:
        save_fig = plt.figure(figsize=(14, 6))
        save_ax_net = save_fig.add_subplot(1, 2, 1)
        save_ax_3d = save_fig.add_subplot(1, 2, 2, projection="3d")

        def save_update(frame_idx: int) -> None:
            frame_data = frames[frame_idx]
            draw_cube_net(save_ax_net, frame_data["net_cube"])
            draw_cube_3d_moving(
                save_ax_3d,
                frame_data["render_cube"],
                frame_data["active_rotation"],
                azim=42,
            )

        ani = animation.FuncAnimation(
            save_fig,
            save_update,
            frames=len(frames),
            interval=interval_ms,
            repeat=False,
        )
        writer = animation.PillowWriter(fps=max(1, 1000 // max(1, interval_ms)))
        ani.save(save_path, writer=writer)
        plt.close(save_fig)
        print(f"Animation saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3x3 Rubik's Cube random scramble + animated solver")
    parser.add_argument("--scramble-length", type=int, default=20, help="Number of random scramble moves")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible scramble")
    parser.add_argument("--interval", type=int, default=500, help="Milliseconds per solving move in animation")
    parser.add_argument("--save", type=str, default=None, help="Optional output GIF path, e.g. solve.gif")
    parser.add_argument("--no-show", action="store_true", help="Run without opening a GUI window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scramble = random_scramble(args.scramble_length, seed=args.seed)
    solve = RubiksCube.inverse_sequence(scramble)

    print("Scramble:", " ".join(scramble))
    print("Solution:", " ".join(solve))

    animate_solution(
        scramble_moves=scramble,
        solve_moves=solve,
        interval_ms=args.interval,
        show=not args.no_show,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
