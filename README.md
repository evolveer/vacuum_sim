# Vacuum Chamber Evacuation Simulation

This repository contains a fully interactive physical simulation of a vacuum chamber pump-down process, built with Python and Streamlit. The simulation calculates pressure decay over time, flow regimes (viscous, transitional, molecular), effective pumping speed, differtent added up Leak Sources and gas throughput based on the specified chamber and pump parameters.

## Prerequisites

To run this simulation, you need to have **Python 3.8 or higher** installed on your system.

## Installation and Running

Follow the instructions below based on your operating system.

### 🪟 Windows

1. **Install Python:**
   If you don't have Python installed, download it from the [official Python website](https://www.python.org/downloads/windows/) and run the installer. **Important:** Make sure to check the box that says "Add Python to PATH" during installation.

2. **Open Command Prompt or PowerShell:**
   Press the Windows key, type `cmd` or `powershell`, and press Enter.

3. **Navigate to the project folder:**
   Use the `cd` command to navigate to the folder where you extracted the `vacuum_sim` files. For example:
   ```cmd
   cd C:\Users\YourName\Downloads\vacuum_sim
   ```

4. **(Optional but recommended) Create a virtual environment:**
   ```cmd
   python -m venv venv
   ```
   Activate the virtual environment:
   ```cmd
   venv\Scripts\activate
   ```

5. **Install required packages:**
   Install the necessary dependencies using `pip`:
   ```cmd
   pip install streamlit plotly scipy numpy
   ```

6. **Run the application:**
   Start the Streamlit server:
   ```cmd
   streamlit run app.py
   ```
   A browser window should automatically open pointing to `http://localhost:8501`.

---

### 🐧 Linux (Ubuntu / Debian / etc.)

1. **Open a terminal.**

2. **Install Python and pip (if not already installed):**
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   ```

3. **Navigate to the project folder:**
   ```bash
   cd /path/to/vacuum_sim
   ```

4. **(Optional but recommended) Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

5. **Install required packages:**
   ```bash
   pip install streamlit plotly scipy numpy
   ```

6. **Run the application:**
   ```bash
   streamlit run app.py
   ```
   A browser window will open, or you can manually navigate to `http://localhost:8501` in your web browser.

---

## Features

* **Interactive Dashboard:** Adjust parameters (volume, pump speed, pipe dimensions, pressures) and see the results instantly.
* **Flow Regimes:** Visualizes the transition between viscous, Knudsen (transitional), and molecular flow based on the Knudsen number.
* **Conductance Analysis:** Shows how pipe conductance changes with pressure.
* **Live Formulas:** Displays the mathematical models used for the current simulation state.
* **Export:** Download the simulation data as a CSV file or export the parameters and results as JSON.

## Project Structure

* `app.py`: The main Streamlit application and user interface.
* `physics.py`: Core physical models and equations (conductance, mean free path, etc.).
* `flow_regimes.py`: Logic for classifying flow regimes.
* `simulation.py`: The numerical integration (ODE solver) for calculating pressure over time.
* `plots.py`: Functions for generating the interactive Plotly charts.
* `utils.py`: Helper functions for unit conversion, validation, and text generation.
