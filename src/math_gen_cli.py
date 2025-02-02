#!/usr/bin/env python3
"""
Unified Cognitive Algebra Interactive Shell
=============================================

This interactive CLI tool integrates our generalized cognitive algebra framework.
It includes:
  • A MathGroup operating on 2×2 matrices (for numerical operations) or on math symbols.
  • Dynamic math formulas computed from matrix invariants (trace & determinant).
  • A simulation daemon that runs in the background—computing combinations and 
    dynamic formulas while not spamming your shell.
  • Additional modules for Galileo-Tensor Tuning, Quantum MIS simulation, and PixelState image glitching.

Type 'help' to list commands. Type 'exit' or 'quit' to leave.
"""

import math
import time
import threading
import json
import os
import cmd
import numpy as np
from PIL import Image
import logging

# Optional: Use Rich for enhanced UI if installed.
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
except ImportError:
    Console = None

# ----------------------------
# Configure Logging (to file only)
# ----------------------------
logging.basicConfig(
    filename="cognitive_algebra.log",
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------
# Fundamental Constants
# ----------------------------
PSI = 44.8
XI = 3721.8
TAU = 64713.97
EPSILON = 0.28082
PHI = (1 + math.sqrt(5)) / 2  # Golden Ratio
PI = math.pi

# ----------------------------
# Global Math Expression and Symbols Mapping
# ----------------------------
DEFAULT_MATH_EXPR = "∀x ∈ ℝ, ∃y ∈ ℝ such that y ≈ ∫ x ∘ dx and ∑_{i=1}^n a_i → ℂ"

axiom_map = {
    "∀": "forall",
    "∃": "exists",
    "∈": "in",
    "∉": "notin",
    "⊆": "subset",
    "⊕": "oplus",
    "∘": "circ",
    "∑": "sum",
    "∏": "prod",
    "∫": "int",
    "∂": "partial",
    "∇": "nabla",
    "≈": "approx",
    "→": "to",
    "⇔": "iff",
    "ℝ": "R",
    "ℤ": "Z",
    "ℕ": "N",
    "ℚ": "Q",
    "ℂ": "C"
}
code_map = {v: k for k, v in axiom_map.items()}

# ----------------------------
# MathGroup on 2×2 Matrices or Symbols
# ----------------------------
def default_axiom_operation(a, b):
    if a == "e":
        return b
    if b == "e":
        return a
    return a + "_" + b

def reverse_axiom_operation(a, b):
    if a == "e":
        return b
    if b == "e":
        return a
    return b + "_" + a

class MathGroup:
    def __init__(self, elements, operation):
        """
        Initializes an algebraic group on a set of 2×2 matrices or symbols.
        """
        self.elements = elements
        self.operation = operation
        self.identity = self.find_identity()

    def find_identity(self):
        """
        Finds the identity element for the current operation.
        Uses np.allclose for numpy arrays and standard equality for non-arrays.
        """
        for e in self.elements:
            is_identity = True
            for a in self.elements:
                # If both e and a are numpy arrays, use np.allclose.
                if isinstance(e, np.ndarray) and isinstance(a, np.ndarray):
                    if not (np.allclose(self.operation(e, a), a) and np.allclose(self.operation(a, e), a)):
                        is_identity = False
                        break
                else:
                    if not (self.operation(e, a) == a and self.operation(a, e) == a):
                        is_identity = False
                        break
            if is_identity:
                return e
        # Fallback for matrices.
        if any(isinstance(x, np.ndarray) for x in self.elements):
            if self.operation == matrix_addition:
                return np.zeros((2,2))
            elif self.operation == matrix_multiplication:
                return np.identity(2)
        return "e"

    def has_inverses(self):
        """
        Checks whether every element has an inverse.
        (This design intentionally does not guarantee inverses.)
        """
        if self.identity is None:
            return False
        for a in self.elements:
            found_inverse = False
            for b in self.elements:
                # Use np.allclose if both are arrays.
                if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                    if np.allclose(self.operation(a, b), self.identity) and np.allclose(self.operation(b, a), self.identity):
                        found_inverse = True
                        break
                else:
                    if self.operation(a, b) == self.identity and self.operation(b, a) == self.identity:
                        found_inverse = True
                        break
            if not found_inverse:
                return False
        return True

    def redefine_operation(self, new_operation):
        """
        Dynamically redefines the group operation (simulating a paradigm shift).
        """
        self.operation = new_operation
        self.identity = self.find_identity()
        if not self.has_inverses():
            logger.warning("The new operation disrupts group properties (inverses missing).")
        else:
            logger.info("Group operation redefined successfully with all properties intact.")

    def combine(self, a, b):
        """
        Combines two elements using the current binary operation.
        """
        return self.operation(a, b)

# ----------------------------
# Matrix Operations and Generators
# ----------------------------
def matrix_addition(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b

def matrix_multiplication(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.matmul(a, b)

def generate_matrices():
    """
    Generates a fixed list of 2×2 matrices built from our fundamental constants.
    """
    M1 = np.array([[PSI, XI], [TAU, EPSILON]])
    M2 = np.array([[PHI, PI], [EPSILON, TAU]])
    M3 = np.array([[XI, PSI], [EPSILON, PHI]])
    M4 = np.array([[TAU, EPSILON], [PI, XI]])
    M5 = np.array([[EPSILON, TAU], [PSI, PHI]])
    return [M1, M2, M3, M4, M5]

def dynamic_math_formula(iteration: int, matrix_result: np.ndarray) -> str:
    """
    Generates a dynamic math formula using the trace and determinant of a matrix.
    """
    tr = np.trace(matrix_result)
    det = np.linalg.det(matrix_result)
    mod_factor = math.sin(iteration / 10.0)
    return f"Trace: {tr:.2f}, Det: {det:.2f}  =>  {tr:.2f} + {mod_factor:.2f} * {det:.2f}"

def create_math_axiom_group():
    symbols = {char for char in DEFAULT_MATH_EXPR if char in axiom_map}
    symbols.add("e")
    return MathGroup(symbols, default_axiom_operation)

# ----------------------------
# Simulation Daemon (Matrix MathGroup)
# ----------------------------
class SimulationDaemon(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.matrices = generate_matrices()
        self.group = MathGroup(self.matrices, matrix_addition)
        self.current_op = "addition"
        self.iteration = 0
        self.index = 0
        self.running = False
        self.data_lock = threading.Lock()
        self.data = []  # Store simulation results.

    def run(self):
        self.running = True
        n = len(self.matrices)
        while self.running:
            with self.data_lock:
                self.iteration += 1
                a = self.matrices[self.index % n]
                b = self.matrices[(self.index + 1) % n]
                self.index += 1
                result = self.group.combine(a, b)
                formula = dynamic_math_formula(self.iteration, result)
                self.data.append({
                    "iteration": self.iteration,
                    "operation": self.current_op,
                    "combined": result.tolist(),
                    "formula": formula
                })
                if self.iteration % 5 == 0:
                    new_op = matrix_multiplication if self.current_op == "addition" else matrix_addition
                    self.group.redefine_operation(new_op)
                    self.current_op = "multiplication" if self.current_op == "addition" else "addition"
                    self.data.append({
                        "iteration": self.iteration,
                        "paradigm_shift": True,
                        "new_identity": self.group.identity.tolist() if isinstance(self.group.identity, np.ndarray) else self.group.identity,
                        "has_inverses": self.group.has_inverses()
                    })
            time.sleep(1)

    def stop(self):
        self.running = False

    def dump_data(self, filepath="simulation_data.jsonl"):
        with self.data_lock:
            with open(filepath, "w") as f:
                for entry in self.data:
                    f.write(json.dumps(entry) + "\n")
            logger.info(f"Simulation data dumped to {filepath}")

    def get_dashboard_table(self):
        table = Table(title="Simulation Dashboard")
        table.add_column("Iteration", justify="right", style="cyan")
        table.add_column("Op", style="magenta")
        table.add_column("Combined (Matrix)", style="green")
        table.add_column("Dynamic Formula", style="yellow")
        with self.data_lock:
            for entry in self.data[-10:]:
                if entry.get("paradigm_shift"):
                    table.add_row(
                        str(entry["iteration"]),
                        "[bold red]Shift[/bold red]",
                        f"New ID: {entry['new_identity']}",
                        f"Inverses? {entry['has_inverses']}"
                    )
                else:
                    table.add_row(
                        str(entry["iteration"]),
                        entry["operation"],
                        str(entry["combined"]),
                        entry["formula"]
                    )
        return table

# Global simulation daemon instance.
sim_daemon = SimulationDaemon()

# ----------------------------
# Additional Modules
# ----------------------------
def galileo_transform(frequency: float) -> float:
    """
    Applies a Galileo-Tensor transformation to a frequency.
    """
    v = np.array([frequency, frequency / PHI, frequency / (PHI ** 2), 1.0])
    T = np.array([
        [PSI, EPSILON, 0.0, PI],
        [EPSILON, XI, TAU, 0.0],
        [0.0, TAU, PI, EPSILON],
        [PI, 0.0, EPSILON, PSI]
    ])
    transformed = T.dot(v)
    return np.linalg.norm(transformed)

def tuning_simulation() -> str:
    base = 440.0
    results = (
        f"Base transformed: {galileo_transform(base):.4f} Hz\n"
        f"Perfect Fifth (1.5×): {galileo_transform(base * 1.5):.4f} Hz\n"
        f"Octave (2.0×): {galileo_transform(base * 2.0):.4f} Hz\n"
    )
    return results

def quantum_simulation(time_point: float) -> str:
    phase = TAU * math.sin(time_point / 10.0)
    scale = PSI + EPSILON * math.cos(time_point / 15.0)
    output = scale * math.cos(phase)
    return f"At t = {time_point:.2f}, Quantum MIS output: {output:.4f}"

def glitch_image(input_path: str, output_path: str, block_size: int) -> str:
    if not os.path.isfile(input_path):
        return f"File not found: {input_path}"
    img = Image.open(input_path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    h, w, _ = arr.shape
    out_arr = arr.copy()
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            factor = 1 + 0.05 * math.sin((x + y) / 20.0)
            out_arr[y:y+block_size, x:x+block_size, :] *= factor
    out_arr = np.clip(out_arr, 0, 255).astype(np.uint8)
    Image.fromarray(out_arr, "RGB").save(output_path)
    return f"Glitched image saved to: {output_path}"

# ----------------------------
# Interactive Shell using cmd
# ----------------------------
class CognitiveAlgebraCLI(cmd.Cmd):
    intro = "Welcome to the Cognitive Algebra Interactive Shell. Type 'help' to list commands.\n"
    prompt = "CognitiveAlgebra> "
    console = Console() if Console else None

    def do_show_expr(self, arg):
        "Display original, compressed, and decompressed math expressions."
        original = DEFAULT_MATH_EXPR
        compressed = " ".join(axiom_map.get(char, char) for char in original)
        decompressed = "".join(code_map.get(token, token) for token in compressed.split())
        output = f"Original:\n{original}\n\nCompressed:\n{compressed}\n\nDecompressed:\n{decompressed}\n"
        self.stdout.write(output)

    def do_math_status(self, arg):
        "Show current MathAxiomGroup status."
        group = create_math_axiom_group()
        elements = sorted(group.elements)
        output = f"MathAxiomGroup Status:\nElements: {elements}\nIdentity: {group.identity}\nHas Inverses: {group.has_inverses()}\n"
        self.stdout.write(output)

    def do_start_sim(self, arg):
        "Start the simulation daemon."
        if not sim_daemon.running:
            sim_daemon.start()
            self.stdout.write("Simulation daemon started.\n")
        else:
            self.stdout.write("Simulation daemon is already running.\n")

    def do_stop_sim(self, arg):
        "Stop the simulation daemon."
        if sim_daemon.running:
            sim_daemon.stop()
            self.stdout.write("Stopping simulation daemon...\n")
            time.sleep(1)
            self.stdout.write("Simulation daemon stopped.\n")
        else:
            self.stdout.write("Simulation daemon is not running.\n")

    def do_dump_data(self, arg):
        "Dump simulation data to a file. Usage: dump_data [filepath]"
        filepath = arg.strip() if arg.strip() else "simulation_data.jsonl"
        sim_daemon.dump_data(filepath)
        self.stdout.write(f"Data dumped to {filepath}\n")

    def do_dashboard(self, arg):
        "Display a live dashboard of simulation data (press Ctrl+C to exit)."
        if not sim_daemon.running:
            self.stdout.write("Simulation daemon is not running. Use 'start_sim' first.\n")
            return
        try:
            while True:
                table = sim_daemon.get_dashboard_table()
                if self.console:
                    self.console.clear()
                    self.console.print(table)
                else:
                    self.stdout.write(str(table) + "\n")
                time.sleep(1)
        except KeyboardInterrupt:
            self.stdout.write("\nDashboard terminated by user.\n")

    def do_tuning(self, arg):
        "Run the Galileo-Tensor Tuning simulation."
        result = tuning_simulation()
        self.stdout.write("Galileo-Tensor Tuning Simulation:\n" + result + "\n")

    def do_quantum(self, arg):
        "Run the Quantum MIS simulation. Usage: quantum [time-point]"
        try:
            t_point = float(arg.strip()) if arg.strip() else 10.0
        except ValueError:
            self.stdout.write("Please enter a valid number for time-point.\n")
            return
        result = quantum_simulation(t_point)
        self.stdout.write("Quantum MIS Simulation:\n" + result + "\n")

    def do_pixel(self, arg):
        "Glitch an image using PixelState Transform ideas. Usage: pixel <input> <output> [block_size]"
        parts = arg.split()
        if len(parts) < 2:
            self.stdout.write("Usage: pixel <input> <output> [block_size]\n")
            return
        input_path = parts[0]
        output_path = parts[1]
        try:
            block_size = int(parts[2]) if len(parts) >= 3 else 8
        except ValueError:
            self.stdout.write("Block size must be an integer.\n")
            return
        result = glitch_image(input_path, output_path, block_size)
        self.stdout.write(result + "\n")

    def do_full_output(self, arg):
        "Display output from all modules."
        self.do_show_expr(arg)
        self.do_math_status(arg)
        self.do_dump_data(arg)
        self.do_tuning(arg)
        self.do_quantum(arg)
        self.stdout.write("For image glitching, use the 'pixel' command with file paths.\n")
        self.stdout.write("For a live dashboard, use the 'dashboard' command.\n")

    def do_exit(self, arg):
        "Exit the interactive shell."
        return True

    def do_quit(self, arg):
        "Exit the interactive shell."
        return True

    def emptyline(self):
        pass

def create_math_axiom_group():
    symbols = {char for char in DEFAULT_MATH_EXPR if char in axiom_map}
    symbols.add("e")
    return MathGroup(symbols, default_axiom_operation)

if __name__ == "__main__":
    try:
        shell = CognitiveAlgebraCLI()
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\nExiting Cognitive Algebra Interactive Shell. Goodbye!")