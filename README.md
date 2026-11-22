# Hadron Collider Beam Trajectory Surrogate (PINN vs. PI-DeepONet)

Physics-Informed Machine Learning (PIML) surrogate modeling for transverse proton betatron oscillations in a periodic quadrupole magnet lattice governed by **Hill's Equation**:

$$\frac{d^2x}{ds^2} + K_0 \cos(2\pi s)\,x = 0, \qquad s \in [0, 3]$$

This repository implements and benchmarks two SciML paradigms in PyTorch against an 8th-order numerical ODE solver (`scipy.integrate.solve_ivp`, `DOP853`):
1. **Single-Instance PINN:** Solves a fixed magnet and injection configuration using periodic lattice features, a hard-enforced initial-condition gate, and two-stage **Adam + L-BFGS** optimization.
2. **Physics-Informed DeepONet (PI-DeepONet):** A Branch–Trunk operator network that learns the continuous mapping $(x_0, x'_0, K_0) \mapsto x(s)$ across varying collider magnet strengths and injection states.

---

## Benchmark Results

![Final Comparison Benchmark](PIML%20Plot.png)

| Metric | Single-Instance PINN | PI-DeepONet |
| :--- | :---: | :---: |
| **Known Trajectory Error ($K_0 = 10.73$)** | **0.047%** | 1.900% |
| **Unseen Trajectory Error ($K_0 = 9.36$)** | 134.381% *(fails without retrain)* | **2.545%** |
| **Mean Error (100 Unseen Trajectories)** | Fails | **2.240%** *(Median: 1.914%)* |
| **Time to Predict 100 New Trajectories** | ~2,180 s *(per-case retraining)* | **~0.13 s** *(>16,000× speedup)* |

---

## Key Technical Takeaways
* **Mitigating Trivial-Zero Collapse:** Because Hill's equation is homogeneous, a naive PINN collapsed to $x(s) \to 0$ (94.8% error). Adding periodic input features ($\cos(2\pi s), \sin(2\pi s)$), a bounded initial-condition gate $(1 - e^{-s})^2$, and L-BFGS optimization reduced single-instance error to **0.047%**.
* **Operator Generalization & Phase-Space Fidelity:** While a standard PINN requires full retraining whenever magnet focusing gradients change, PI-DeepONet generalizes across unseen $(x_0, x'_0, K_0)$ configurations in milliseconds while preserving $(x, dx/ds)$ phase-space orbits.

## Repository Contents
* [`PIMLproject.ipynb`](PIMLproject.ipynb) — End-to-end Google Colab notebook (ground-truth `DOP853` simulator, PINN, PI-DeepONet, and evaluation plots).
* [`EE5203 Project Proposal - Particle Trajectory Prediction in a Hadron Collider.pdf`](EE5203%20Project%20Proposal%20-%20Particle%20Trajectory%20Prediction%20in%20a%20Hadron%20Collider.pdf) — Project proposal outlining the accelerator physics formulation and experimental design.
