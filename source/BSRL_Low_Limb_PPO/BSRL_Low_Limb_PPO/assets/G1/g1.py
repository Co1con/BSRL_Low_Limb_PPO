"""Paths to the Unitree G1 model files bundled with this project."""

from pathlib import Path


G1_MODEL_DIR = Path(__file__).resolve().parent
G1_23DOF_USD = str(G1_MODEL_DIR / "23dof/usd/g1_23dof_rev_1_0/g1_23dof_rev_1_0.usd")
G1_29DOF_USD = str(G1_MODEL_DIR / "29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd")
