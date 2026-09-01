# src/architectures/extractor.py
import os
import torch
import numpy as np
from numpy.polynomial.chebyshev import chebfit
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.architectures.hep_kan import HEPKAN


class SymbolicWarmStartExtractor:
    """
    Extractor de warm-start que preserva la ESTRUCTURA del grafo clásico
    (nodos suma vs. nodos multiplicación, y qué aristas confluyen en cada uno),
    en vez de aplanar la red a una lista plana de inputs activos.

    IMPORTANTE - de dónde lee: este extractor está pensado para correr sobre el
    checkpoint 03_retrained (post pruning + retraining, ANTES de la simplificación
    simbólica), y ajusta sus polinomios de Chebyshev contra la rama NUMÉRICA
    (act_fun, splines aprendidos) de cada arista, no contra symbolic_fun. En esa
    etapa symbolic_fun sigue siendo el placeholder cero de pykan (fix_symbolic
    todavía no se ha llamado), así que act_fun es la única rama con información
    real. Esto evita un ajuste-de-un-ajuste: antes se leía symbolic_fun del
    checkpoint 05_final (post simplificación simbólica + fine-tuning), añadiendo
    una aproximación simbólica lossy de más entre la spline aprendida y el
    Chebyshev final.

    Convención de índices (idéntica a la que ya usa HEPKAN.plot(), verificada
    y funcional en tu pipeline):
        - act_fun[l].mask[i][j]        -> [nodo_previo=i][neurona_cruda=j]
        - symbolic_fun[l].mask[j][i]   -> [neurona_cruda=j][nodo_previo=i]
    Una arista (i -> j) en la capa l está activa si act_fun[l].mask[i][j] != 0
    (el mask de act_fun refleja la poda por sí solo en esta etapa del pipeline).

    IMPORTANTE - alcance de esta versión:
    Esta extracción está pensada para arquitecturas de profundidad 2
    (entradas -> capa oculta -> salida), que es tu caso actual
    (width=[22,[9,9],1], depth=2). Generalizar a profundidad >2 en un solo
    circuito coherente requeriría medición intermedia + re-encoding (ver
    discusión previa) — no lo resuelve este extractor.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.degree = self.config.get("chebyshev_degree", 4)
        # Umbral de rango dinámico: una arista con variación insignificante
        # (Δy = max(y) - min(y)) se descarta, igual que antes, pero ahora
        # se aplica arista por arista en TODAS las capas, no solo en la de entrada.
        self.dynamic_range_threshold = self.config.get("qkan_dynamic_range_threshold", 1e-3)

    # ------------------------------------------------------------------
    # Evaluación aislada de una arista (idéntico a tu versión original,
    # ya generalizado por (layer_index, input_index, output_index))
    # ------------------------------------------------------------------
    def _evaluate_isolated_edges(self, model, layer_index, input_index, output_index, x_vals):
        layer_width = model.width[layer_index]
        in_dim = layer_width[0] if isinstance(layer_width, list) else layer_width
        # in_dim real de la capa: usamos width_in para no asumir formato de 'width'
        in_dim = int(model.width_in[layer_index])
        n = len(x_vals)

        x_zero = torch.zeros((n, in_dim), dtype=torch.float32).to(self.device)
        x_var = torch.zeros((n, in_dim), dtype=torch.float32).to(self.device)
        x_var[:, input_index] = torch.tensor(x_vals, dtype=torch.float32).to(self.device)

        def layer_forward(x_in):
            try:
                # Lee la rama NUMERICA (splines) en vez de la simbolica: en el
                # checkpoint 03_retrained, symbolic_fun[l] sigue siendo el
                # placeholder cero de pykan (fix_symbolic aun no se ha llamado),
                # asi que act_fun es la unica rama con informacion real en esta
                # etapa del pipeline.
                numeric = model.act_fun[layer_index](x_in)
                x_out = numeric[0] if isinstance(numeric, tuple) else numeric
            except Exception:
                out_dim = int(model.width_out[layer_index + 1])
                x_out = torch.zeros((n, out_dim), dtype=torch.float32).to(self.device)

            # x_out esta en la dimension RAW pre-colapso por multiplicacion
            # (width_out[l+1]), igual que output_index (ver _build_node_groups /
            # raw_indices). El afin que corresponde a ESE punto del forward real
            # de pykan es subnode_bias/subnode_scale (tamano width_out[l+1]), NO
            # node_bias/node_scale (tamano width_in[l+1], que se aplica DESPUES
            # de la colapsion por multiplicacion). Usar node_bias/node_scale aqui
            # es un bug de forma en cualquier capa con nodos de multiplicacion
            # supervivientes (width_out != width_in), como la capa oculta de la
            # config de produccion (9 suma + 9 mult).
            if hasattr(model, "subnode_bias") and model.subnode_bias is not None and len(model.subnode_bias) > layer_index:
                x_out = x_out + model.subnode_bias[layer_index]
            if hasattr(model, "subnode_scale") and model.subnode_scale is not None and len(model.subnode_scale) > layer_index:
                x_out = x_out * model.subnode_scale[layer_index]
            return x_out

        with torch.no_grad():
            y_var = layer_forward(x_var)[:, output_index].cpu().numpy()
            y_zero = layer_forward(x_zero)[:, output_index].cpu().numpy()

        y_var = np.nan_to_num(y_var, nan=0.0, posinf=0.0, neginf=0.0)
        y_zero = np.nan_to_num(y_zero, nan=0.0, posinf=0.0, neginf=0.0)
        return y_var - y_zero + (y_zero / in_dim)

    def _fit_edge(self, model, layer_index, input_index, output_index, x_vals):
        y_vals = self._evaluate_isolated_edges(model, layer_index, input_index, output_index, x_vals)
        dynamic_range = float(np.max(y_vals) - np.min(y_vals))
        coefs = chebfit(x_vals, y_vals, deg=self.degree)
        return coefs.tolist(), dynamic_range

    # ------------------------------------------------------------------
    # Agrupamiento de neuronas crudas en nodos colapsados (suma / mult),
    # replicando EXACTAMENTE la lógica de agrupamiento que ya usa plot()
    # para dibujar las conexiones entre capas.
    # ------------------------------------------------------------------
    def _build_node_groups(self, model, layer_plus_1_idx):
        """
        Devuelve una lista de nodos colapsados para la capa `layer_plus_1_idx`
        (es decir, la capa DESTINO de las aristas de la capa layer_plus_1_idx-1).
        Cada nodo es un dict: {'type': 'sum'|'mult', 'raw_indices': [...]}
        """
        width = model.width
        width_out = np.array(model.width_out)
        n_sum = width[layer_plus_1_idx][0]
        n_raw_total = int(width_out[layer_plus_1_idx])

        groups = []
        for j in range(n_sum):
            groups.append({"type": "sum", "raw_indices": [j]})

        mult_id = 0
        i = n_sum
        while i < n_raw_total:
            ma = model.mult_arity if isinstance(model.mult_arity, int) else model.mult_arity[layer_plus_1_idx][mult_id]
            raw_idx = list(range(i, i + ma))
            groups.append({"type": "mult", "raw_indices": raw_idx})
            i += ma
            mult_id += 1

        return groups

    # ------------------------------------------------------------------
    # Extracción principal
    # ------------------------------------------------------------------
    def extract_and_save(self, classic_model_path, output_graph_path, report_path):
        print("\n" + "=" * 40)
        print("[Extractor] Extracción de grafo estructurado (suma/mult) para QKAN")
        print("=" * 40)

        base_kan = HEPKAN.loadckpt(classic_model_path)
        model = HEPKAN.__new__(HEPKAN)
        model.__dict__.update(base_kan.__dict__)
        model.to(self.device)
        model.eval()

        x_vals = np.linspace(-1, 1, 500)
        depth = len(model.width) - 1
        if depth != 2:
            print(f"[Extractor] ADVERTENCIA: se esperaba depth=2 (entrada->oculta->salida), "
                  f"se encontró depth={depth}. Esta versión del extractor NO generaliza "
                  f"a más profundidad sin medición intermedia.")

        # ---- Capa 0: entradas -> neuronas crudas ocultas -----------------
        n_inputs = int(model.width_in[0])
        n_raw_hidden = int(model.width_out[1])

        raw_edges_layer0 = {j: [] for j in range(n_raw_hidden)}  # j -> lista de {input_idx, coefs}
        active_inputs_set = set()

        print("[Extractor] Evaluando aristas activas: entradas -> capa oculta...")
        for j in range(n_raw_hidden):
            for i in range(n_inputs):
                # symbolic_fun's mask is always zero at this pipeline stage
                # (03_retrained, pre fix_symbolic) — act_fun's mask alone
                # correctly reflects which edges survived pruning.
                mask_act = model.act_fun[0].mask[i, j].item()
                if mask_act == 0.0:
                    continue

                coefs, dyn_range = self._fit_edge(model, 0, i, j, x_vals)
                if dyn_range <= self.dynamic_range_threshold:
                    continue

                raw_edges_layer0[j].append({"input_idx": i, "coefs": coefs, "dynamic_range": dyn_range})
                active_inputs_set.add(i)

        active_inputs = sorted(active_inputs_set)
        if not active_inputs:
            raise RuntimeError("[Extractor] Ninguna arista de entrada superó el umbral de rango dinámico. "
                                "Revisa 'qkan_dynamic_range_threshold' o la poda clásica.")

        # input_pos: raw_idx clásico -> posición de LECTURA dentro del tensor de datos ya
        # filtrado (self.active_inputs en el modelo). Esto NO es un wire cuántico: un mismo
        # input puede leerse desde aquí varias veces, en varios wires distintos, sin conflicto
        # (fan-out). El wire cuántico (acumulador) se asigna más abajo, uno por cada neurona
        # cruda/grupo que sobrevive la poda — nunca uno por input.
        input_pos = {raw_idx: pos for pos, raw_idx in enumerate(active_inputs)}
        print(f"[Extractor] {len(active_inputs)} variables clásicas activas (inputs raw: {active_inputs})")

        for j in raw_edges_layer0:
            for e in raw_edges_layer0[j]:
                e["col"] = input_pos[e["input_idx"]]  # de dónde se LEE el dato (columna clásica)

        # ---- Agrupar neuronas crudas en nodos colapsados de capa oculta ---
        # Aquí es donde se asignan los WIRES cuánticos reales: uno por cada
        # neurona-cruda/grupo que sobrevive la poda (un acumulador), NUNCA uno
        # por input. Un mismo input puede escribir (vía _qkan_edge) en varios
        # de estos wires si alimenta varias ramas (fan-out) — eso es correcto
        # y esperado, no un error.
        hidden_groups = self._build_node_groups(model, 1)  # nodos de la capa 1 (oculta)
        hidden_nodes = []
        wire_counter = 0
        for node in hidden_groups:
            edge_groups = []
            has_any_edge = False
            for raw_j in node["raw_indices"]:
                edges = raw_edges_layer0.get(raw_j, [])
                if edges:
                    has_any_edge = True
                    group_wire = wire_counter
                    wire_counter += 1
                    for e in edges:
                        e["wire"] = group_wire  # wire DEDICADO a este acumulador (no al input)
                edge_groups.append(edges)
            if not has_any_edge:
                # Nodo oculto totalmente podado (ninguna de sus neuronas crudas sobrevivió)
                continue
            hidden_nodes.append({"type": node["type"], "edge_groups": edge_groups})

        n_qubits = wire_counter
        n_sum_survivors = sum(1 for h in hidden_nodes if h["type"] == "sum")
        n_mult_survivors = sum(1 for h in hidden_nodes if h["type"] == "mult")
        print(f"[Extractor] Nodos ocultos sobrevivientes: {n_sum_survivors} suma, {n_mult_survivors} multiplicación")
        print(f"[Extractor] {n_qubits} qubits requeridos (uno por acumulador/neurona-cruda sobreviviente, "
              f"NO uno por input — puede diferir del número de variables clásicas activas)")

        # ---- Capa 1: nodos ocultos colapsados -> salida -------------------
        n_hidden_collapsed = int(model.width_in[1])
        n_output_raw = int(model.width_out[2])  # normalmente 1

        # Mapear índice de nodo oculto "colapsado" (0..n_hidden_collapsed-1) -> posición en hidden_nodes
        # (el orden de _build_node_groups ya coincide con el índice colapsado real de pykan)
        collapsed_to_hidden = {}
        collapsed_idx = 0
        kept_idx = 0
        for node in hidden_groups:
            # necesitamos saber si este nodo sobrevivió (está en hidden_nodes) para mapearlo
            raw_j0 = node["raw_indices"][0]
            survived = any(raw_edges_layer0.get(rj, []) for rj in node["raw_indices"])
            if survived:
                collapsed_to_hidden[collapsed_idx] = kept_idx
                kept_idx += 1
            collapsed_idx += 1

        output_edges = []
        print("[Extractor] Evaluando aristas activas: capa oculta -> salida...")
        for out_j in range(n_output_raw):
            for h_collapsed in range(n_hidden_collapsed):
                mask_act = model.act_fun[1].mask[h_collapsed, out_j].item()
                if mask_act == 0.0:
                    continue
                if h_collapsed not in collapsed_to_hidden:
                    continue  # el nodo oculto que alimentaba esta arista ya fue podado del todo

                coefs, dyn_range = self._fit_edge(model, 1, h_collapsed, out_j, x_vals)
                if dyn_range <= self.dynamic_range_threshold:
                    continue

                output_edges.append({
                    "hidden_idx": collapsed_to_hidden[h_collapsed],
                    "coefs": coefs,
                    "dynamic_range": dyn_range,
                })

        if not output_edges:
            raise RuntimeError("[Extractor] Ninguna arista oculta->salida sobrevivió el umbral. "
                                "Revisa la poda de nodos/aristas de la capa 1.")

        print(f"[Extractor] Aristas oculta->salida activas: {len(output_edges)}")

        graph = {
            "n_qubits": n_qubits,
            "active_inputs": active_inputs,   # índices RAW clásicos (para filtrar X en forward())
            "degree": self.degree,
            "hidden_nodes": hidden_nodes,      # lista de {type, edge_groups: [[{wire,coefs}, ...], ...]}
            "output_edges": output_edges,      # lista de {hidden_idx, coefs}
        }

        os.makedirs(os.path.dirname(output_graph_path), exist_ok=True)
        torch.save(graph, output_graph_path)
        print(f"[Extractor] Grafo cuántico exportado a: {output_graph_path}")

        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write("=== REPORTE DE EXTRACCIÓN ESTRUCTURADA (GRAFO) ===\n")
            f.write(f"n_qubits: {n_qubits}\n")
            f.write(f"active_inputs (raw): {active_inputs}\n")
            f.write(f"Nodos ocultos sobrevivientes: {len(hidden_nodes)} "
                    f"({n_sum_survivors} suma, {n_mult_survivors} mult)\n")
            f.write(f"Aristas oculta->salida: {len(output_edges)}\n\n")
            for idx, node in enumerate(hidden_nodes):
                n_edges = sum(len(g) for g in node["edge_groups"])
                f.write(f"  Nodo oculto {idx} [{node['type']}]: {len(node['edge_groups'])} grupo(s), "
                        f"{n_edges} arista(s) totales\n")

        return graph