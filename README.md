# CLI

### a cli tool to try out advanced computation operations


> deps: `pip install pyyaml click numpy matplotlib pillow`

> to use clone the project or copy src/cli.py


## Introduction
This code suite implements a **classical HPC** (High Performance Computing) approach, borrowing inspiration from quantum theoretical concepts (e.g., Morphotensorial Field Theory, doping-layers, multi-qudit states, and random HPC noise). It does not run on actual quantum hardware. Instead, it simulates “quantum-like” processes to produce results that might interest:

1. **Professionals** who want HPC concurrency and doping-layers toggles for fast experiments.
2. **Students** exploring quantum-inspired or HPC-based methods, but who don’t have quantum hardware access.

The code’s major goals include:
- Running **param sweeps** (dimensions & protection levels) with HPC concurrency.  
- Generating **glitch-art images** by applying HPC doping-based transformations to input pictures.  
- Testing **various noise models** (Gaussian, 1/f, random_telegraph) at user-specified doping strengths and protection levels.

---

## Core Principles

1. **Quantum-Like HPC**  
   - We treat each qudit (multi-level quantum unit) in HPC memory, performing random gates and partial decoherence.  
   - \(\psi=44.8,\xi=3721.8,\tau=64713.97,\epsilon=0.28082,\phi=1.61803\) are approximate constants from a theoretical framework.  
   - The HPC doping-layers offset or scale coherence times, gating random transformations.

2. **Doping-Layers**  
   - Each doping layer modifies HPC “coherence” or “energy.” You can specify doping strengths, doping correlation distance, etc.  
   - HPC doping is purely a classical approach to randomizing or scaling parameters used in the code’s “quantum” logic.

3. **Noise Models**  
   - *Gaussian*: Adds normal-distributed perturbations to HPC gates.  
   - *1/f (flicker)*: Reduces noise amplitude as the “time parameter” grows.  
   - *Random Telegraph*: Occasionally flips sign (±) in HPC gates.  

4. **Protection Levels**  
   - An integer 1–5, each specifying a HPC environment’s approximate “error correction.”  
   - Higher levels might reduce HPC decoherence, but you might still see surprising outcomes if doping or dimension is large.

---

## Usage Overview

1. **Install Dependencies**  
   - Python 3, plus `numpy`, `matplotlib`, `PIL (Pillow)`, `click`, and `yaml`.

2. **CLI Commands**  
   The `cli.py` script has multiple subcommands. Each subcommand addresses a distinct HPC scenario:

   **a) `show-constants`**  
   - Prints fundamental HPC constants (ψ, ξ, τ, ε, φ).  
   - Optionally merges doping-layers offsets from a YAML config if `--config`.

   **b) `inject-data`**  
   - Reads doping or HPC environment offsets from a YAML/JSON file, storing them in the session-level dictionary.  
   - You can keep doping data in a separate file (e.g., `doping_layers.yaml`) and load it once.

   **c) `run-sim`**  
   - The main HPC simulation for single or multi-qudit runs.  
   - `--multi` toggles multi-qudit approach; `--size` picks how many HPC qudits.  
   - `--trials` sets the random gating iterations.  
   - The output is a single line like `Fidelity => 0.310`.

   **d) `parallel-sweep`**  
   - Param sweep for HPC dimension ∈ `[3..n]` and protection level ∈ `[3..m]`.  
   - You define these combos in a YAML (like `big_sweep.yaml`), including concurrency, doping, and noise.  
   - Results are shown line by line, then optionally saved to JSON.

   **e) `plot-coherence`**  
   - Takes a config, calculates the five HPC protection levels (1–5), and plots their HPC-based coherence times.  
   - Saved to `coherence_plot.png`.

   **f) `ml-optimize`**  
   - Minimal HPC-based machine learning approach.  
   - Repeats HPC runs for a given number of episodes, searching for the highest fidelity found.  
   - Good for automated HPC tuning or “best possible approach” within random gates.

   **g) `distribute-sweep`**  
   - Splits dimension–protection combos among HPC “clusters” or nodes.  
   - Doesn’t do concurrency itself; simply prints distribution of combos.  
   - For HPC use if you want to manually assign tasks across multiple nodes.

   **h) `glitch-image`**  
   - A visually interesting subcommand that applies HPC doping-layers logic and random HPC gates to each block of an image, producing “glitch-art.”  
   - Example usage:
     ```bash
     python cli.py glitch-image \
        -i input.jpg -o out.png \
        --dimension 4 --protection 5 \
        --block-size 8 --noise-model random_telegraph \
        --doping 0.07 --randomize
     ```
   - Lower dimension and protection might produce subtle changes, while higher dimension or doping randomization can yield extreme glitch patterns.

---

## Example Scenarios

1. **Basic Single Qudit**  
   ```bash
   python cli.py run-sim --config my_run.yaml --trials 100
   ```
   This loads HPC config from `my_run.yaml`, sets dimension/protection from it, and does 100 random gating cycles. The final fidelity is displayed as a float.

2. **Parallel Sweep**  
   ```bash
   python cli.py parallel-sweep --config big_sweep.yaml
   ```
   Suppose `big_sweep.yaml` enumerates `param_dims: [3,4,5,6]` and `param_prots: [3,4,5]`. This command calculates HPC fidelity for each dimension–protection pair, printing lines like:
   ```
   Dim=3 Prot=3 Fidelity=0.325
   ...
   ```
   The results may be saved to a JSON file if `param_output` is set in the YAML.

3. **Image Glitch**  
   ```bash
   python cli.py glitch-image -i cityscape.jpg -o glitched_city.png -d 4 -p 3 --block-size 8 --noise-model=1/f
   ```
   - Splits the image into 8×8 pixel blocks, each block viewed as HPC qudit dimension=4, protection=3.  
   - Applies HPC doping-layers and gates.  
   - Saves final glitch art to `glitched_city.png`.

---

## Reading the Results

1. **Fidelity**  
   - The code’s HPC “fidelity” is the fraction of times the HPC measure yields outcome=0. Higher means fewer HPC-like errors.  
   - Large dimension or doping can degrade fidelity, especially at moderate HPC protection.

2. **Plot Coherence**  
   - You see how doping and HPC temperature reduce or scale your HPC coherence times across levels 1–5. If doping is intense, the HPC lines can drastically shift downward.

3. **Glitch Image**  
   - Larger dimension, stronger doping randomization => more intense color shifts.  
   - People can visually see HPC-based random gating as glitch patterns.

---

## Advice for Professionals

- **HPC Concurrency**  
  If your HPC cluster or local 8-core machine is powerful, set `concurrency: 8` in your YAML so `parallel-sweep` spawns 8 processes. This drastically cuts run time for big param sweeps.

- **Slurm**  
  If you set `use_slurm: true`, the script can generate a `.slurm` file for HPC queue submission. Tweak `job_name`, `slurm_time`, `slurm_nodes`, `slurm_ntasks`.

- **ML Pipeline**  
  For HPC-coded random approaches, `ml-optimize` is a start. You could integrate RL or other HPC-based ML steps, building on the minimal structure provided.

---

## Advice for Students

- **Learn HPC**  
  This tool teaches concurrency and HPC param sweeps in a quantum-inspired context—**not** actual quantum computing, but the code structure is realistic for HPC tasks.

- **Experiment**  
  - Swap noise models (`gaussian`, `1/f`, `random_telegraph`) to see how HPC fidelity changes.  
  - Adjust doping in a YAML to watch how HPC doping-layers shift results.

- **Explore the Glitch Command**  
  - Even outside HPC concurrency, the glitch subcommand is a creative gateway to see HPC doping-layers produce chaotic transformations on images.


The HPC quantum-inspired CLI is a **classical** toolset that merges doping-layers, random HPC gating, concurrency, and minimal error correction. Professionals can harness concurrency to run broad param sweeps or doping-layers explorations on HPC clusters, while students can enjoy toggling dimension/protection for moderate local runs, possibly generating glitch art. The code’s fidelity outputs, doping-layers logic, advanced noise, and image transformations unify into a single HPC environment that’s flexible yet best used with an understanding of HPC concurrency and (simulated) quantum-like processes.

Below is a **complete engineering breakdown** of your code, focusing on its architecture, key components, and functionality. I’ll ignore trivialities (e.g., basic Python syntax, standard library imports) and focus on the core engineering aspects.

---

## **1. Core Architecture**
The code is structured as a **command-line interface (CLI)** tool with a modular design. It uses the following key components:
- **CLI Framework**: Built using the `click` library for command-line argument parsing.
- **Simulation Core**: Implements quantum-inspired HPC simulations using classes like `HPCConfig`, `HPCMFT`, `HPCQudit`, and `HPCMultiQuditRegister`.
- **Noise Models**: Implemented in the `HPCAdvancedNoise` class.
- **Data Management**: Handles session data, logging, and file I/O using dictionaries and JSON/YAML files.
- **Concurrency**: Supports multiprocessing for parallel parameter sweeps and Slurm integration for HPC clusters.
- **Visualization**: Generates plots and glitch-art images using `matplotlib` and `PIL`.

---

## **2. Key Components**

### **2.1 Constants and Session Data**
- **Global Constants**:  
  - `PSI`, `XI`, `TAU`, `EPSILON`, `PHI`, `BOLTZMANN`, `PI` are theoretical constants used in the simulation.
  - These constants define the behavior of the quantum-inspired HPC system.

- **Session Data**:  
  - `SESSION_DATA`: A dictionary storing runtime configurations like `psi_offset`, `xi_scale`, `coherence_mult`, and `layer_params`.
  - This allows dynamic adjustments to the simulation without modifying the code.

---

### **2.2 Configuration (`HPCConfig`)**
- **Purpose**:  
  - Encapsulates all simulation parameters (e.g., dimension, protection level, noise model, doping settings).
  - Acts as a central configuration object passed to other components.

- **Key Parameters**:  
  - `dimension`: Dimension of the qudit (e.g., 3 for qutrit, 4 for ququad).
  - `protection_level`: Error correction level (1–5).
  - `doping_layers`: List of doping layers with concentrations and strengths.
  - `noise_model`: Type of noise (`gaussian`, `1/f`, `random_telegraph`).
  - `temperature`: Thermal energy scale for noise calculations.
  - `concurrency`: Number of parallel processes for parameter sweeps.
  - `use_slurm`: Enables Slurm integration for HPC clusters.

---

### **2.3 Morphotensorial Field Theory (`HPCMFT`)**
- **Purpose**:  
  - Implements the theoretical framework for quantum-inspired HPC simulations.
  - Calculates energy levels, coherence times, and doping effects.

- **Key Methods**:  
  - `_calc_levels`: Computes protection levels, energy, and coherence times based on doping and noise.
  - Uses constants like `PSI`, `XI`, `TAU`, and `EPSILON` to derive theoretical values.

---

### **2.4 Noise Models (`HPCAdvancedNoise`)**
- **Purpose**:  
  - Simulates noise in the HPC system, affecting gate operations and coherence.

- **Key Methods**:  
  - `thermal_noise_factor`: Computes thermal noise based on Boltzmann's constant and temperature.
  - `apply_gate_noise`: Applies noise to gate matrices based on the selected noise model:
    - **Gaussian**: Adds normal-distributed noise.
    - **1/f (flicker)**: Reduces noise amplitude over time.
    - **Random Telegraph**: Randomly flips the sign of noise.

---

### **2.5 Qudit Simulation (`HPCQudit`)**
- **Purpose**:  
  - Represents a single qudit (multi-level quantum unit) in the HPC simulation.

- **Key Methods**:  
  - `apply_gate`: Applies a noisy gate operation to the qudit's state (`rho`).
  - `measure`: Simulates a measurement, collapsing the qudit's state.
  - `_maybe_decohere`: Simulates decoherence over time.
  - `_apply_errors`: Applies errors based on the qudit's error rate.

- **State Representation**:  
  - The qudit's state is represented by a density matrix (`rho`), initialized to the ground state (`|0⟩⟨0|`).

---

### **2.6 Multi-Qudit Simulation (`HPCMultiQuditRegister`)**
- **Purpose**:  
  - Represents a register of multiple qudits for parallel simulations.

- **Key Methods**:  
  - `apply_gate_to_all`: Applies random gates to all qudits in the register.
  - `run_surface_code_checks`: Performs surface code error correction on all qudits.
  - `measure_all`: Measures all qudits and returns the results.

---

### **2.7 CLI Commands**
The CLI is built using the `click` library and supports the following commands:

#### **a) `show-constants`**
- **Functionality**:  
  - Prints theoretical constants and optionally loads doping offsets from a YAML config.

#### **b) `inject-data`**
- **Functionality**:  
  - Injects doping or HPC environment offsets from a YAML/JSON file into the session data.

#### **c) `run-sim`**
- **Functionality**:  
  - Runs single or multi-qudit simulations and outputs fidelity.

#### **d) `parallel-sweep`**
- **Functionality**:  
  - Performs parameter sweeps over dimensions and protection levels.
  - Supports multiprocessing and Slurm integration.

#### **e) `plot-coherence`**
- **Functionality**:  
  - Plots coherence times for protection levels 1–5.

#### **f) `ml-optimize`**
- **Functionality**:  
  - Repeats HPC runs to find the highest fidelity, simulating a minimal ML pipeline.

#### **g) `distribute-sweep`**
- **Functionality**:  
  - Splits dimension–protection combos across clusters or nodes.

#### **h) `glitch-image`**
- **Functionality**:  
  - Applies HPC doping and random gates to image blocks, producing glitch art.

---

### **2.8 Concurrency and Slurm Integration**
- **Multiprocessing**:  
  - The `parallel-sweep` command uses the `multiprocessing.Pool` class to parallelize parameter sweeps.
  - Each process runs a simulation for a specific dimension–protection combo.

- **Slurm Integration**:  
  - The `slurm_submit` function generates a Slurm script for HPC cluster submission.
  - Parameters like `job_name`, `slurm_time`, `slurm_nodes`, and `slurm_ntasks` are customizable.

---

### **2.9 Data Management and Logging**
- **Session Data**:  
  - Stored in the `SESSION_DATA` dictionary and updated via CLI commands.
  - Persisted to JSON/YAML files for reuse.

- **Logging**:  
  - Simulation results and logs are saved to JSONL files in the `research_data` directory.
  - Each log entry includes a timestamp and relevant metadata.

---

### **2.10 Visualization**
- **Coherence Plot**:  
  - Generated using `matplotlib` and saved as `coherence_plot.png`.

- **Glitch Art**:  
  - Created by applying HPC doping and random gates to image blocks using the `PIL` library.

---

## **3. Workflow**
1. **Configuration**:  
   - Users define simulation parameters in a YAML file or via CLI arguments.
   - Doping layers, noise models, and protection levels are configured.

2. **Simulation**:  
   - The `run-sim` or `parallel-sweep` command executes the simulation.
   - Qudits are initialized, gates are applied, and measurements are performed.

3. **Output**:  
   - Fidelity results are printed or saved to JSON files.
   - Coherence plots and glitch-art images are generated.

---

## **4. Engineering Highlights**


- **Extensibility**:  
  - New noise models, doping effects, or CLI commands can be added without disrupting existing functionality.

- **Performance**:  
  - Multiprocessing and Slurm integration enable efficient parameter sweeps on HPC clusters.

- **Visualization**:  
  - The `glitch-image` command provides a creative application of the simulation logic.

