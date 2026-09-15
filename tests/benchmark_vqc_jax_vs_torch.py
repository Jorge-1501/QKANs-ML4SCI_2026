# Ad-hoc scratch benchmark (not pytest, no assertions) -- run directly with
# `python tests/benchmark_vqc_jax_vs_torch.py`. Times one Adam training step
# on a small synthetic VQC shaped like a real extracted QKAN graph (see
# reports/AUC_test/qkan_chebyshev_degree_comparison.md for observed real
# graph sizes: 4-7 qubits), comparing the repo's current production path
# (PennyLane lightning.qubit + torch, CPU-only) against an equivalent
# PennyLane default.qubit + JAX circuit (CPU and, if usable, GPU).
#
# Requires `jax` to be installed (not a repo dependency -- see
# reports/benchmarks/vqc_jax_vs_torch_training_step.md for the
# install/uninstall procedure used to produce this report without adding it
# to pyproject.toml/uv.lock). Does not modify src/architectures/qkan_model.py
# -- the torch/lightning variant constructs the real, unmodified QKANModel;
# the JAX variant is a hand-written line-for-line port of the same circuit.
import argparse
import contextlib
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.architectures.qkan_model import QKANModel  # noqa: E402

SEED = 0
N_QUBITS = 6
DEGREE = 4
N_FEATURES = 7
BATCH_SIZE = 32
N_WARMUP = 3
N_TIMED = 8
LR = 5e-3
ADAM_B1 = 0.9
ADAM_B2 = 0.999
ADAM_EPS = 1e-8


@contextlib.contextmanager
def _suppress_stdout():
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


def build_synthetic_graph_dict(seed=SEED):
    """A 6-qubit graph mixing a sum node (2 edges, free accumulation) and two
    mult nodes (arity 2 and 3, single vs. chained IsingZZ) -- shaped exactly
    like QKANModel.__init__ expects (src/architectures/qkan_model.py:40-133),
    matching the qubit-count range and sum/mult mix seen in real extracted
    graphs, but with arbitrary weight values since this is a timing test."""
    rng = np.random.default_rng(seed)

    def coefs():
        return rng.uniform(-0.5, 0.5, DEGREE + 1).tolist()

    return {
        "n_qubits": N_QUBITS,
        "active_inputs": list(range(N_FEATURES)),
        "degree": DEGREE,
        "hidden_nodes": [
            {
                "type": "sum",
                "edge_groups": [[
                    {"col": 0, "wire": 0, "coefs": coefs()},
                    {"col": 1, "wire": 0, "coefs": coefs()},
                ]],
            },
            {
                "type": "mult",
                "edge_groups": [
                    [{"col": 2, "wire": 1, "coefs": coefs()}],
                    [{"col": 3, "wire": 2, "coefs": coefs()}],
                ],
            },
            {
                "type": "mult",
                "edge_groups": [
                    [{"col": 4, "wire": 3, "coefs": coefs()}],
                    [{"col": 5, "wire": 4, "coefs": coefs()}],
                    [{"col": 6, "wire": 5, "coefs": coefs()}],
                ],
            },
        ],
        "output_edges": [
            {"hidden_idx": 0, "coefs": coefs()},
            {"hidden_idx": 1, "coefs": coefs()},
            {"hidden_idx": 2, "coefs": coefs()},
        ],
    }


def build_synthetic_dataset(seed=SEED):
    rng = np.random.default_rng(seed + 1)
    X = rng.uniform(-1.0, 1.0, size=(BATCH_SIZE, N_FEATURES)).astype(np.float32)
    y = rng.integers(0, 2, size=(BATCH_SIZE,)).astype(np.float32)
    return X, y


def _build_plan(graph):
    """Pure-Python port of QKANModel.__init__'s topology-building loop
    (qkan_model.py:63-112), so the JAX circuit's connectivity is provably
    identical to what QKANModel itself would build from the same graph dict."""
    edge_table = []
    zz_table = []
    hidden_final_wire = []
    output_table = []

    for node in graph["hidden_nodes"]:
        raw_carrier_wires = []
        for group in node["edge_groups"]:
            if not group:
                continue
            acc_wire = group[0]["wire"]
            for edge in group:
                assert edge["wire"] == acc_wire
                edge_table.append({
                    "in_col": edge["col"],
                    "acc_wire": acc_wire,
                    "coefs": edge["coefs"],
                })
            raw_carrier_wires.append(acc_wire)

        if not raw_carrier_wires:
            continue

        if node["type"] == "mult" and len(raw_carrier_wires) > 1:
            base = raw_carrier_wires[0]
            for other in raw_carrier_wires[1:]:
                zz_table.append({"wire_a": other, "wire_b": base})
            hidden_final_wire.append(base)
        else:
            hidden_final_wire.append(raw_carrier_wires[0])

    output_wire = hidden_final_wire[0] if hidden_final_wire else 0

    for oe in graph["output_edges"]:
        src_wire = hidden_final_wire[oe["hidden_idx"]]
        output_table.append({"src_wire": src_wire, "coefs": oe["coefs"]})

    return edge_table, zz_table, output_table, output_wire


def _step_stats(timed_steps):
    return {
        "mean_step_seconds": statistics.mean(timed_steps),
        "median_step_seconds": statistics.median(timed_steps),
        "std_step_seconds": statistics.pstdev(timed_steps),
        "min_step_seconds": min(timed_steps),
        "max_step_seconds": max(timed_steps),
        "steps_per_second": 1.0 / statistics.mean(timed_steps),
    }


def run_torch_benchmark(graph, X_np, y_np):
    with tempfile.TemporaryDirectory() as td:
        graph_path = os.path.join(td, "quantum_weights.pt")
        torch.save(graph, graph_path)
        with _suppress_stdout():
            model = QKANModel(graph_path=graph_path, backend_mode="ideal")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, betas=(ADAM_B1, ADAM_B2), eps=ADAM_EPS
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    def train_step():
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()

    first_call, warmups, timed, final_loss = None, [], [], None
    for i in range(N_WARMUP + N_TIMED):
        t0 = time.perf_counter()
        final_loss = train_step()
        dt = time.perf_counter() - t0
        if i == 0:
            first_call = dt
        elif i < N_WARMUP:
            warmups.append(dt)
        else:
            timed.append(dt)

    return {
        "framework": "torch",
        "device": "cpu",
        "status": "ok",
        "error": None,
        "n_qubits": model.n_qubits,
        "n_trainable_params": sum(p.numel() for p in model.parameters()),
        "batch_size": BATCH_SIZE,
        "n_warmup": N_WARMUP,
        "n_timed": N_TIMED,
        "first_call_seconds": first_call,
        "warmup_seconds": warmups,
        "timed_step_seconds": timed,
        "final_loss": final_loss,
        **_step_stats(timed),
    }


def run_jax_benchmark(graph, X_np, y_np, expected_platform):
    import jax
    import jax.numpy as jnp
    import pennylane as qml

    edge_table, zz_table, output_table, output_wire = _build_plan(graph)
    degree = graph["degree"]
    n_qubits = graph["n_qubits"]
    dev = qml.device("default.qubit", wires=n_qubits)

    def circuit(edge_weights, zz_weights, output_weights, x):
        for idx, e in enumerate(edge_table):
            theta = jnp.arccos(jnp.clip(x[:, e["in_col"]], -0.9999, 0.9999))
            w = edge_weights[idx]
            for i in range(degree):
                qml.RY(w[i], wires=e["acc_wire"])
                qml.RZ(theta, wires=e["acc_wire"])
            qml.RY(w[degree], wires=e["acc_wire"])
        for idx, zz in enumerate(zz_table):
            qml.IsingZZ(zz_weights[idx], wires=[zz["wire_a"], zz["wire_b"]])
            qml.CNOT(wires=[zz["wire_a"], zz["wire_b"]])
        for idx, oe in enumerate(output_table):
            if oe["src_wire"] == output_wire:
                qml.RY(output_weights[idx][0], wires=output_wire)
            else:
                qml.IsingZZ(output_weights[idx][0], wires=[oe["src_wire"], output_wire])
                qml.CNOT(wires=[oe["src_wire"], output_wire])
        return qml.expval(qml.PauliZ(output_wire))

    qnode = qml.QNode(circuit, dev, interface="jax")

    edge_weights = jnp.array([e["coefs"] for e in edge_table], dtype=jnp.float32)
    zz_weights = jnp.zeros(len(zz_table), dtype=jnp.float32)
    output_weights = jnp.array([o["coefs"] for o in output_table], dtype=jnp.float32)
    params = {
        "edge_weights": edge_weights,
        "zz_weights": zz_weights,
        "output_weights": output_weights,
    }
    n_trainable_params = int(
        edge_weights.size + zz_weights.size + output_weights.size
    )

    X = jnp.array(X_np, dtype=jnp.float32)
    y = jnp.array(y_np, dtype=jnp.float32)

    def bce_with_logits(z, y):
        return jnp.mean(jnp.maximum(z, 0) - z * y + jnp.log1p(jnp.exp(-jnp.abs(z))))

    def loss_fn(params, X, y):
        out = qnode(params["edge_weights"], params["zz_weights"], params["output_weights"], X)
        return bce_with_logits(out, y)

    def adam_init(params):
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        return {"m": zeros, "v": jax.tree_util.tree_map(jnp.zeros_like, params), "t": jnp.array(0.0)}

    def adam_update(params, grads, state):
        t = state["t"] + 1.0
        m = jax.tree_util.tree_map(lambda m_, g: ADAM_B1 * m_ + (1 - ADAM_B1) * g, state["m"], grads)
        v = jax.tree_util.tree_map(lambda v_, g: ADAM_B2 * v_ + (1 - ADAM_B2) * g * g, state["v"], grads)
        m_hat = jax.tree_util.tree_map(lambda m_: m_ / (1 - ADAM_B1 ** t), m)
        v_hat = jax.tree_util.tree_map(lambda v_: v_ / (1 - ADAM_B2 ** t), v)
        new_params = jax.tree_util.tree_map(
            lambda p, mh, vh: p - LR * mh / (jnp.sqrt(vh) + ADAM_EPS), params, m_hat, v_hat
        )
        return new_params, {"m": m, "v": v, "t": t}

    def train_step(params, state, X, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, X, y)
        new_params, new_state = adam_update(params, grads, state)
        return new_params, new_state, loss

    train_step_jit = jax.jit(train_step)
    state = adam_init(params)

    first_call, warmups, timed, final_loss = None, [], [], None
    for i in range(N_WARMUP + N_TIMED):
        t0 = time.perf_counter()
        params, state, loss = train_step_jit(params, state, X, y)
        loss = jax.block_until_ready(loss)
        dt = time.perf_counter() - t0
        final_loss = float(loss)
        if i == 0:
            first_call = dt
        elif i < N_WARMUP:
            warmups.append(dt)
        else:
            timed.append(dt)

    devices = jax.devices()
    seen_platform = devices[0].platform if devices else "unknown"
    status = "ok" if seen_platform == expected_platform else "fallback_to_cpu"

    return {
        "framework": "jax",
        "device": expected_platform,
        "status": status,
        "error": None,
        "n_qubits": n_qubits,
        "n_trainable_params": n_trainable_params,
        "batch_size": BATCH_SIZE,
        "n_warmup": N_WARMUP,
        "n_timed": N_TIMED,
        "first_call_seconds": first_call,
        "warmup_seconds": warmups,
        "timed_step_seconds": timed,
        "final_loss": final_loss,
        "jax_devices": [str(d) for d in devices],
        **_step_stats(timed),
    }


def dispatch_variant(variant):
    graph = build_synthetic_graph_dict()
    X_np, y_np = build_synthetic_dataset()
    try:
        if variant == "torch":
            return run_torch_benchmark(graph, X_np, y_np)
        elif variant == "jax_cpu":
            return run_jax_benchmark(graph, X_np, y_np, expected_platform="cpu")
        elif variant == "jax_gpu":
            return run_jax_benchmark(graph, X_np, y_np, expected_platform="gpu")
        else:
            raise ValueError(f"Unknown variant: {variant}")
    except Exception:
        return {
            "framework": variant,
            "status": "failed",
            "error": traceback.format_exc()[-2000:],
        }


def run_variant_subprocess(variant):
    env = os.environ.copy()
    if variant == "torch":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif variant == "jax_cpu":
        env["JAX_PLATFORMS"] = "cpu"
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif variant == "jax_gpu":
        env["JAX_PLATFORMS"] = "cuda"
        env.pop("CUDA_VISIBLE_DEVICES", None)

    try:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--variant", variant],
            env=env, capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        return {"framework": variant, "status": "failed", "error": f"timeout: {exc}"}

    stdout = (proc.stdout or "").strip()
    if not stdout:
        return {
            "framework": variant,
            "status": "failed",
            "error": (proc.stderr or "no stdout produced")[-2000:],
        }
    try:
        return json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError:
        return {
            "framework": variant,
            "status": "failed",
            "error": f"non-JSON stdout: {stdout[-2000:]} | stderr: {(proc.stderr or '')[-1000:]}",
        }


def _pkg_version(dist_name):
    import importlib.metadata
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_meta():
    gpu_name = None
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            gpu_name = out.stdout.strip()
    except Exception:
        pass

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "gpu": gpu_name,
        "pennylane_version": _pkg_version("pennylane"),
        "pennylane_lightning_version": _pkg_version("pennylane-lightning"),
        "torch_version": _pkg_version("torch"),
        "jax_version": _pkg_version("jax"),
        "n_qubits": N_QUBITS,
        "chebyshev_degree": DEGREE,
        "batch_size": BATCH_SIZE,
        "n_warmup": N_WARMUP,
        "n_timed": N_TIMED,
        "graph_topology": "1 sum node (2 edges) + 1 mult node (arity 2) + 1 mult node (arity 3)",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["all", "torch", "jax_cpu", "jax_gpu"], default="all")
    args = parser.parse_args()

    if args.variant != "all":
        with _suppress_stdout():
            result = dispatch_variant(args.variant)
        print(json.dumps(result))
        return

    combined = {"meta": build_meta()}
    for variant in ("torch", "jax_cpu", "jax_gpu"):
        combined[variant] = run_variant_subprocess(variant)
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
