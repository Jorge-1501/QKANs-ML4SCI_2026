# Quantum Sinusoidal Kolmogorov-Arnold Networks for High Energy Physics: Project Summary

## 1. Overview and Motivation

This document summarizes the work carried out for **QKAN**, a Google Summer of Code 2026 project at ML4SCI. It follows the three project notebooks (`EDA_top.ipynb`, `Training_process.ipynb` and `Results.ipynb`) and concentrates on what each of them concludes and contributes.

Variational quantum circuits (VQCs) are limited in size, because classical simulation cost grows as $O(2^Q)$ with the number of qubits $Q$. Feeding a jet with dozens of features into such a circuit is therefore impractical. We use a classical **Kolmogorov-Arnold Network (KAN)** as a **pre-processor and qubit filter**. The KAN is trained, pruned to a small and interpretable topology, and then distilled into a quantum circuit that inherits its structure. The number of surviving hidden nodes decides the number of qubits, so the classical model determines the quantum resources instead of leaving that choice to trial and error.

The workflow preprocesses the jets into balanced replicates, trains and prunes the KAN, extracts each surviving edge into a compact basis (Chebyshev or sine) as a **warm start**, fine-tunes the QKAN on ideal, shot-based and noisy simulators, and compares everything against a classical Random Forest.

The main task is **top-quark tagging**, using the public dataset from [Zenodo record 2603256](https://zenodo.org/records/2603256). Quark-gluon and Higgs datasets are referenced in the repository, but only top tagging is developed end to end.

---

## 2. Data Exploration and Preprocessing (`EDA_top.ipynb`)

### 2.1 What the dataset contains

The dataset consists of Monte Carlo jets simulated at 14 TeV with Pythia8 and a Delphes ATLAS detector card, without pile-up or multi-parton interactions. Jets are clustered with anti-$k_T$ at $R = 0.8$ in the $p_T$ range [550, 650] GeV. The leading 200 constituents are stored as four-momenta, zero-padded, and sorted by $p_T$. Signal jets (top quarks) are labeled 1 and QCD background 0.

The HDF5 file does not document its column order, so we inferred it from the data. The column means show a pattern every four columns: the first column of each block has a much larger mean than the other three, which stay near zero. We therefore concluded that each block is one constituent, with the **energy first and the three momentum components after it**.

### 2.2 Physical observations

From the distributions of the reconstructed quantities we drew four conclusions.

- **Invariant mass discriminates but is not sufficient.** Signal peaks around the top-quark mass (about 173 GeV), while the background is broader. The two overlap, so we select the window **145-205 GeV** for the main experiments. Inside this window the trivial mass cue is largely removed and the classifiers must rely on subtler information.
- **Top jets are more populated.** The multiplicity distribution of top jets sits at higher values and is wider than that of QCD jets. This is consistent with a three-body decay.
- **Top jets are more diffuse.** The radial energy profile and the cumulative $p_T$ fraction rise more gradually for tops and more steeply for QCD, so QCD energy is more concentrated near the jet axis.
- **The $\eta$-$\phi$ scatter plots are not decisive.** We expected a visible difference in dispersion, but noise and axis scales hide it.

We also found an artifact: the $\Delta R$ distribution shows no sharp cut at 0.8, and its outliers come from the small constant added to avoid division by zero. Constituents beyond the jet radius are therefore filtered out.

### 2.3 Feature engineering

The processed representation combines **global** and **local** features:

- **Global:** the jet mass $m_{jet} = \sqrt{E^2 - p_x^2 - p_y^2 - p_z^2}$ and the jet multiplicity, counting constituents with energy above $10^{-8}$.
- **Local:** for each retained constituent, the distance to the jet axis $\Delta R$ and the relative transverse momentum $p_{T,rel}$.

To choose how many constituents to keep, we used the cumulative $p_T$ distribution and an 80% criterion, which suggested about 15 constituents. The training pipeline finally uses the **first 10 constituents**, giving **22 inputs** (2 global and 10 pairs of local features). This truncation limits model size without discarding most of the physics.

Local features already lie in $(0, 1)$. For the global features we applied a logarithmic transformation and then a **tanh normalization**. We deliberately kept the outliers, because the centers of the two class distributions are similar and the tails carry information.

### 2.4 Balancing and replicates

The number of events in the mass window differs between classes, with more tops. We **undersample the majority class** so that both classes have equal representation. The balanced dataset is then split into **5 disjoint, class-balanced subsets**. Each seed selects one subset (`seed % 5`), so several seeds provide independent end-to-end replicates instead of a single point estimate. Preprocessing writes a canonical cache once, and every later script only selects from it.

---

## 3. Training Process and Architecture (`Training_process.ipynb`)

### 3.1 The classical KAN

The classical model is built on [pykan](https://github.com/kindxiaoming/pykan) and modified in `HEPKAN`. The modifications are motivated by the available computational resources:

- A **bug fix in `prune_input`**, which passed a module instead of a name string when rebuilding the pruned model and broke checkpoint serialization.
- A **plotting routine** that reuses one Matplotlib figure for all edges and skips pruned edges, reducing time and memory on wide networks.
- A **no-op history logger**, which avoids writing a checkpoint on every model mutation during pruning and symbolic search.

The architecture is `[22, [9, 9], 1]`. The input layer has 22 features, the hidden layer offers up to 9 addition nodes and 9 multiplication nodes, and the output has a single unit. The B-splines use grid size 5 and order $k = 3$, for **8,568 parameters** in total.

The **multiplication nodes** matter because the original Kolmogorov-Arnold theorem only guarantees a representation through univariate functions and addition. Multiplication lets the network express feature interactions directly.

For the reference run (seed 10) the base model reached a test AUC of **0.792** and stopped early after about 20 of the 60 planned epochs.

### 3.2 Pruning, retraining and symbolic fitting

Pruning removes inputs below a threshold of 0.01, then hidden nodes and edges with attribution thresholds of 0.04 and 0.06. A hard **fan-in cap** keeps only the two strongest input edges per hidden neuron. The pruned structure is then retrained for 20 epochs, during which the validation AUC rose from 0.714 to about 0.754.

The notebook also covers **symbolic fitting**. Each surviving edge is matched against a library of functions, which yields a formula-level interpretation of the model. We consider this the main interpretability advantage of KANs. It belongs to the classical branch; the quantum branch instead takes the numerical response of each edge.

### 3.3 Extraction to a quantum graph

The extractor isolates the response of each active edge by disconnecting the other inputs of its target node. It exports a serialized graph of sum and multiplication nodes. In the reference run, the surviving structure was small:

- **2 active input variables** (feature indices 0 and 1, the jet mass and the multiplicity),
- **1 sum node and 5 multiplication nodes** in the hidden layer,
- **11 qubits**, 18 input edges, 5 `IsingZZ` transfers and 6 output edges.

Qubits are counted per surviving accumulator node, **not per input**. The same input can be re-uploaded onto several wires when it feeds several hidden nodes.

### 3.4 The quantum circuit

The circuit follows five design principles:

1. **Data re-uploading** models each univariate function through repeated rotations of the input.
2. The hidden layer decides which features matter and how many qubits are used.
3. **Addition is free**: consecutive $R_Z$ rotations on the same wire accumulate their angles, so sum nodes need no two-qubit gate.
4. **Multiplication** uses an `IsingZZ` gate combined with a `CNOT`.
5. The information is collapsed onto one output wire, and the prediction is a single-qubit Pauli-Z expectation.

The hidden-to-output stage is a variational readout and not a literal second KAN layer, because a hidden node's value lives in a qubit phase and cannot be re-uploaded without mid-circuit measurement. Only depth-2 networks are therefore supported.

The model can run on three simulators: `ideal` (`lightning.qubit`), `shots` (finite-shot `default.qubit`) and `noisy` (Qiskit Aer with a noise model derived from `FakeManilaV2`). Training uses binary cross-entropy with logits, the Adam optimizer and a `ReduceLROnPlateau` scheduler that halves the learning rate on a validation plateau. Because simulation is expensive, each epoch trains on a **fresh random subset** (about 1,000 samples) and validates on a fixed subset.

---

## 4. Warm-Start Strategies

We compare three ways of initializing the circuit angles.

**Chebyshev polynomials.** Each isolated edge response is fitted with $y \approx \sum_{i=0}^{N} c_i T_i(x)$ on $[-1, 1]$, and the coefficients become the initial rotation angles.

*Degree selection.* The degree is fixed at $N = 4$ for every edge. An earlier version searched for the smallest degree that met an $R^2$ threshold. On smaller training pools this search almost always selected low degrees, and the circuit lost part of the information of the classical structure. The effect appeared as an abrupt drop in baseline AUC (from about 0.80 to 0.26-0.36), and reverting to the fixed degree restored it.

**Sine basis.** As an alternative we implemented a fixed-frequency sinusoidal basis, $y \approx \sum_k A_k \sin(\text{freq}_k x + \text{phase}_k)$, with amplitudes obtained by least squares. Its fit is moderate, with a **mean $R^2$ of about 0.49-0.61**, well below Chebyshev. Without a constant term it cannot represent static offsets and produces negative $R^2$ on some edges. An extended basis with a constant term is proposed as future work.

**Random initialization.** As a control, we keep the pruned topology exactly (same qubits and connections) but sample every angle from $\mathcal{N}(0, 1)$. This isolates the value of the classical knowledge from the value of the topology alone.

---

## 5. Random Forest Baseline

To calibrate what is achievable classically, we add a **Random Forest** with 500 trees and balanced class weights. In contrast to the KAN pipeline, it receives all **22 features** without pruning. Its metrics use the same keys as the KAN trainers, so all models can be aggregated into one table.

The feature importance (MDI/Gini) points to the jet mass and multiplicity as the most predictive variables. This **retrospectively supports the aggressive pruning** of the KAN, which independently retained the same two variables.

---

## 6. Results (`Results.ipynb`)

All per-run metrics are collected into a single Parquet table (74 rows across 6 seeds). We analyze two regimes.

### 6.1 Mass cut and five subsets (seeds 10-14)

Mean test AUC $\pm$ standard deviation over five seeds:

| Model | Test AUC |
|---|---|
| Random Forest (22 features) | 0.823 $\pm$ 0.004 |
| Classical KAN, base | 0.789 $\pm$ 0.002 |
| Classical KAN, after prune + fine-tune | 0.770 $\pm$ 0.014 |
| Classical KAN, retrained / symbolic | 0.756 $\pm$ 0.004 |
| **QKAN, trained (ideal)** | **0.736 $\pm$ 0.006** |
| QKAN, trained (shots / noisy) | 0.732 / 0.730 ($\pm$ 0.006) |
| QKAN warm start only, Chebyshev (ideal) | 0.698 $\pm$ 0.004 |
| QKAN warm start only, Sine (ideal) | 0.637 $\pm$ 0.015 |
| QKAN warm start only, random (ideal) | 0.488 $\pm$ 0.021 |

Three observations follow from this table.

First, pruning and symbolic simplification cost roughly 0.03 AUC relative to the base KAN, and the trained QKAN sits a further 0.02 below the retrained classical model. It reaches an AUC of about 0.73-0.74 with only two input variables and 11 qubits.

Second, **fine-tuning matters**. Training raises the Chebyshev warm start from 0.698 to 0.736 in the ideal case.

Third, ideal, shot-based and noisy backends differ by no more than about 0.006 AUC. Within our noise model, the circuit is not visibly degraded.

**Accuracy requires caution.** The quantum models reach an accuracy of only about 0.56-0.57, compared with roughly 0.71 for the classical KAN. In the reference baseline the confusion matrix collapses toward the positive class (recall 0.98, precision 0.53). The ranking signal, measured by AUC, therefore survives, but the fixed 0.5 decision threshold is poorly calibrated for the circuit output. We report AUC as the primary metric for this reason.

### 6.2 Do the warm starts matter? Hypothesis tests

Since the seeds are few, we tested the differences explicitly with paired t-tests at $\alpha = 0.05$. The null hypothesis was that random initialization and the warm start give the same accuracy and AUC.

| Comparison | Accuracy | AUC |
|---|---|---|
| Random vs Chebyshev | $t = -4.86$, $p = 0.0082$ | $t = -20.17$, $p = 3.6 \times 10^{-5}$ |
| Random vs Sine | $t = -5.00$, $p = 0.0075$ | $t = -17.95$, $p = 5.7 \times 10^{-5}$ |

Both null hypotheses are rejected. We nevertheless read these results cautiously, because they rest on **only five seeds**.

The same ordering appears at the 50% signal-efficiency working point. All models reach the target efficiency, but their background efficiency differs:

- **Chebyshev:** background efficiency 0.183 $\pm$ 0.005, background rejection **5.46 $\pm$ 0.14**.
- **Sine:** background efficiency 0.307 $\pm$ 0.015, background rejection **3.26 $\pm$ 0.15**.
- **Random:** background efficiency 0.560 $\pm$ 0.064, background rejection **1.81 $\pm$ 0.21**.

### 6.3 Almost the full dataset, without mass cut

In the second regime we use nearly all events (with only 10 constituents per jet and no mass cut) as a single block, seed 42. There are no replicates here, so we cannot compute error bars or t-tests, and the results should be read as **a single run**.

| Model | Test AUC |
|---|---|
| Random Forest | 0.965 |
| Classical KAN, base | 0.959 |
| Classical KAN, after prune + fine-tune | 0.959 |
| Classical KAN, retrained / symbolic | 0.953 |
| QKAN, trained (ideal / noisy) | 0.904 / 0.902 |
| QKAN warm start, Chebyshev (ideal / noisy) | 0.708 / 0.677 |

Without the mass cut, the classifiers can exploit the jet mass directly, so every model reaches its highest discrimination, and the trained QKAN keeps an AUC above 0.90. Its accuracy is again lower (0.72-0.74, recall 0.97, precision about 0.65), repeating the calibration issue seen above.

---

## 7. Conclusions

We draw five conclusions from the notebooks.

1. **Classical pruning works as a qubit filter.** The KAN reduces 22 inputs to two variables and 11 qubits, and the circuit still reaches an AUC of about 0.73 (mass cut) and 0.90 (full data).
2. **The warm start carries real information.** The ordering Chebyshev > Sine > Random is consistent across seeds and statistically significant on five replicates. Random initialization performs at chance level (AUC about 0.49), which shows that the topology alone is not enough. The Chebyshev basis is better because its per-edge fit is much closer to the classical response ($R^2$ close to 1 against 0.49-0.61 for sines).
3. **Simulated noise has a small effect.** The gap between the `ideal` and `noisy` backends is at most about 0.006 AUC in the mass-cut regime and about 0.001 after training in the full-data regime. This is encouraging, but it is a simulated noise model and **not a hardware result**.
4. **The mass cut is the harder problem.** With it, AUC drops from about 0.96 to 0.82 for the Random Forest and from 0.90 to 0.73 for the QKAN. Removing the simple mass cue forces the models to rely on substructure, which two surviving variables can only partly capture.
5. **The classical ceiling is set by the Random Forest.** It has access to all features and leads in both regimes. The quantum model does not surpass it. The value of the hybrid approach is a compact, interpretable circuit that preserves a large fraction of that performance, not a gain in accuracy.

---

## 8. Contributions

- **A reproducible, regime-aware workflow.** Data regimes are encoded in the directory layout, replicate subsets are class-balanced, every stage is idempotent by checkpoint, and results from several seeds are collected into one Parquet table.
- **`HEPKAN`**, a pykan subclass that fixes a serialization bug in input pruning and reduces plotting and logging overhead.
- **A pruning rule tailored to quantum limits**, combining attribution thresholds with a hard fan-in cap.
- **A classical-to-quantum bridge.** The extractor turns a pruned KAN into a sum/multiplication graph, and the QKAN builder turns that graph into a PennyLane circuit.
- **Two interchangeable bases plus a control**, which allow the value of the warm start to be measured, on three simulation backends evaluated before and after training.
- **A diagnosed and fixed failure.** The adaptive Chebyshev degree search silently collapsed baseline AUC on smaller data pools; we found the cause and replaced it by a fixed degree.
- **A classical benchmark**, the Random Forest, evaluated with the same metrics and split.
- **An exploratory analysis** documenting the dataset layout, the $\Delta R$ artifact and the choice of 10 constituents and 22 inputs.

---

## 9. Limitations and Future Work

- **Few replicates.** The statistical tests use five seeds, and the full-data regime is a single run without uncertainty.
- **Extreme compression.** Pruning leaves about two input variables, and loosening this raises the qubit count beyond what the simulator can evaluate in reasonable time.
- **Calibration.** Quantum accuracy and precision are weak even where AUC is good. A tuned decision threshold should be studied.
- **Sine basis.** It needs a constant term or more frequencies before it can be judged fairly.
- **Simulated noise only.** `FakeManilaV2` approximates a real device but is not one.
- **Depth-2 networks only.** Deeper KANs would need mid-circuit measurement and re-encoding.
- **Other datasets.** Quark-gluon tagging has a preprocessing pipeline, and Higgs detection is only referenced.

The results show that a classical KAN can determine the structure of a quantum circuit, that the knowledge transferred through an accurate basis has a measurable effect, and that the resulting compact model preserves a substantial part of the classification signal. The model does not yet match the best classical baseline; this limitation is reported together with the results.
