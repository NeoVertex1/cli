# cli
## A cli to learn advanced computing concepts

**HPC Quantum-Inspired CLI: Concise Overview of Commands and Purpose**

---

## Project Overview
This repository contains a **high-performance computing (HPC)–focused, quantum-inspired command-line interface (CLI)**. It simulates multi-level (qudit) systems, doping-layers for HPC parameter offsets, concurrency, and minimal noise/error correction. It does **not** run on quantum hardware—**all processes run classically**—but leverages quantum-like concepts for param sweeps, doping-layers tests, HPC concurrency, and glitch-art image transformations.

### Who Is This For?
- **HPC Practitioners**: Those wanting to try classical concurrency and doping-based offset approaches in a single package.
- **Quantum-Inspired Researchers**: People who want to mimic quantum gating or multi-level states on HPC clusters without specialized quantum equipment.
- **Students or Hobbyists**: Anyone interested in quantum-like HPC simulations or doping-layers. The glitch command is also a fun, immediate visual payoff.

---

## Key Commands and Expected Outcomes

Below are the primary subcommands. All commands can parse a YAML config (e.g. `--config config.yaml`), which might define doping, concurrency, dimension, etc. If the command supports `--verbose`, enabling it will store logs in `research_data/` without overwriting previous results.

1. **`show-constants`**  
   - Prints core HPC constants (`PSI`, `XI`, `TAU`, `EPSILON`, `PHI`).
   - Optionally merges doping-layers from a config file.
   - **Expected Outcome**:  
     ```
     PSI=44.8 XI=3721.8 TAU=64713.97 EPS=0.28082 PHI=1.61803
     Disclaimer: HPC classical simulation only
     ```
   - Use `--verbose` to append session data to `verbose_log.jsonl`.

2. **`inject-data -f <file>`**  
   - Reads doping or HPC offsets from YAML/JSON.
   - Merges them into a session dictionary, printing final session data in JSON.
   - **Expected Outcome**:  
     ```
     {
       "psi_offset": ...,
       "xi_scale": ...,
       "coherence_mult": ...,
       "layer_params": [...],
       "extra_data": {...}
     }
     ```
   - Appends a log entry if verbose.

3. **`run-sim --config <file>`**  
   - Loads HPC config from `<file>` (dimension, protection, doping, concurrency, etc.).
   - Performs either single-qudit or multi-qudit HPC runs (if `--multi`).
   - `--size <N>` to set multi-qudit register size, `--trials <M>` for gating cycles.
   - **Expected Outcome**:  
     ```
     Fidelity => 0.310
     ```
   - If `--verbose`, logs `run_sim_log.jsonl` with all relevant run details.

4. **`parallel-sweep --config <file>`**  
   - Typical HPC concurrency approach. The config can define param sets for dimension & protection arrays (`param_dims`, `param_prots`), plus concurrency threads (`concurrency`).
   - Iterates combos and runs HPC tasks. If concurrency >1, uses multiprocessing.
   - **Expected Outcome** (line by line):
     ```
     Dim=3 Prot=3 Fidelity=0.325
     Dim=3 Prot=4 Fidelity=0.280
     ...
     ```
   - If `param_output: big_sweep_results.json`, merges new combos with old content to avoid overwriting.

5. **`plot-coherence --config <file>`**  
   - Reads HPC doping-layers from config, calculates 5 HPC protection levels, plots their final coherence times, and saves `coherence_plot.png`.
   - **Expected Outcome**:  
     ```
     Saved => coherence_plot.png
     ```
   - Use `--verbose` to log the actual data arrays to `plot_coherence_log.jsonl`.

6. **`ml-optimize --config <file>`**  
   - Minimal HPC-based “learning.” Repeats single-qudit runs for a specified number of episodes (`ml_episodes`).
   - Tries to find the best HPC fidelity encountered.
   - **Expected Outcome**:
     ```
     ML best_fidelity= 0.800, best_info={'episode': 10}
     ```
   - If verbose, logs results in `ml_optimize_log.jsonl`.

7. **`distribute-sweep --clusters <C> --dims <d1,d2,...> --prots <p1,p2,...>`**  
   - Splits dimension × protection combos among `<C>` HPC nodes (no actual concurrency here, just prints distribution).
   - **Expected Outcome**:
     ```
     Node 0 => [(3,3),(3,4),(3,5)]
     Node 1 => [(4,3),(4,4),(4,5)]
     ```
   - Logs if verbose.

8. **`glitch-image -i <input.jpg> -o <output.png>`**  
   - Reads HPC dimension & protection from arguments (or config). Splits the image into blocks (`--block-size N`), applying HPC doping-layers & random gating. 
   - The HPC measure modifies the block intensities, creating glitch-art.
   - **Expected Outcome**:
     ```
     Glitched image saved => out.png
     ```
   - If verbose, logs to `glitch_image_log.jsonl`.

---

## Additional Points

1. **No Overwrite**  
   - JSON line-append format ensures you never lose old HPC results. Each command can be repeated with different doping-layers or concurrency, building a large data repository in `research_data/`.

2. **HPC Concurrency**  
   - `parallel-sweep` is the most HPC-centric. You can set `concurrency: 8` in config to exploit 8 CPU cores. For cluster usage, adapt the script or use the minimal Slurm script generation.

3. **Disclaimer**  
   - This is classical HPC code, **not** hooking into quantum hardware. The doping-layers concept is purely HPC logic, and “protection levels” here are HPC-coded illusions of quantum error correction.

4. **Who Gains Value**  
   - HPC scientists wanting a single tool that runs param sweeps, doping-based modifications, concurrency, plus immediate glitch-art for demonstration.  
   - Students exploring “quantum-like” HPC gating or doping-layers, with minimal overhead.

---

## Concluding Remarks
This HPC quantum-inspired CLI stands out for:

- **Rich HPC Parameterization** (dimension, doping-layers, concurrency, machine learning loops).  
- **Comprehensive Logging** that appends data for further HPC analytics.  
- **Glitch-Art** as a quick visual demonstration of HPC gating logic.  

Professionals can incorporate doping-layers in HPC concurrency sweeps and store results incrementally. Students can see direct HPC illusions of “quantum gating” or produce glitch-art. Overall, the HPC quantum-inspired CLI meets a wide range of educational and HPC-lab usage scenarios, all from the comfort of a single set of subcommands.
