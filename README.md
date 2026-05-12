# MassAwareArm

MAE C263C Project — Team 8 Armstrong: Autonomous Mass-Aware Sorting System.

A UR5e arm in MuJoCo picks colored cubes, infers each cube's mass **without a force sensor or scale**, and sorts them into bins by weight.

- Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Build plan: [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)

## Setup

Requires **Python 3.11** and **Git**.

### 1. Create the virtual environment

```bash
py -3.11 -m venv .env
```

### 2. Activate it

- **Windows (PowerShell):** `.env\Scripts\Activate.ps1`
- **Windows (bash / Git Bash):** `source .env/Scripts/activate`
- **macOS / Linux:** `source .env/bin/activate`

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify

```bash
python -c "import mujoco; print(mujoco.__version__)"
python -m mujoco.viewer --mjcf=software/assets/scene.xml
```

The viewer should open with the UR5e on a table, three colored cubes, and two bin sites.

## Repository layout

```
docs/                      design + build documentation
software/
├── assets/                MuJoCo scene + vendored UR5e + umi_gripper
├── massaware/             python package (perception, estimators, planner, ...)
├── configs/               YAML configs
└── scripts/               run / calibrate / compare_estimators
requirements.txt
.env/                      virtual env (gitignored)
```
