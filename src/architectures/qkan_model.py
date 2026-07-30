# src/architectures/qkan_model.py
import pennylane as qml
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt


class QKANModel(nn.Module):
    """
    QKAN con:
      - n_qubits = número de ACUMULADORES (neuronas-crudas/grupos de la capa
        oculta que sobreviven la poda), NO el número de inputs activos. Un
        input puede re-subirse en varios wires distintos si alimenta varias
        ramas; un wire nunca se identifica con "el input tal".
      - Fan-out resuelto por re-uploading: la misma variable clásica se
        vuelve a subir (RY/RZ) en cada wire-acumulador donde participe, leída
        siempre de su propia columna de datos (in_col), nunca de un wire
        "propio" que no existe.
      - Suma dentro de un nodo resuelta GRATIS: aplicar RZ(theta) repetidas
        veces sobre el MISMO wire acumula los ángulos (rotaciones sobre el
        mismo eje se componen aditivamente) -> no requiere compuertas de 2
        qubits para sumar.
      - Multiplicación resuelta con IsingZZ, y SOLO entre los wires que el
        grafo clásico podado marca como confluyentes en un nodo-mult real
        (la profundidad/cantidad de IsingZZ es ahora dinámica, no fija).

    LIMITACIÓN EXPLÍCITA (no la esconde este código):
    La etapa oculta->salida ("stage 2") no puede re-subir el valor de un
    nodo oculto con una nueva codificación DRU exacta, porque ese valor
    vive en la fase/rotación acumulada de un qubit, no como número clásico
    legible sin medir. Lo que SÍ logra esta versión es que la conectividad
    de esa segunda etapa (qué wires se combinan y con qué peso) se derive
    del grafo real podado, en vez de ser fija como antes. Sigue siendo una
    capa de lectura variacional, no composición funcional literal de dos
    capas KAN. Preservar eso exactamente requeriría medición intermedia +
    re-encoding (ver discusión previa sobre el paper QKAN).
    """

    def __init__(self, graph_path, backend_mode="ideal"):
        super().__init__()

        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"No se encontró el grafo cuántico en {graph_path}. Ejecuta el extractor primero.")

        graph = torch.load(graph_path, weights_only=False)
        self.n_qubits = graph["n_qubits"]
        self.active_inputs = graph["active_inputs"]
        self.degree = graph["degree"]

        # ------------------------------------------------------------
        # Construcción del PLAN ESTÁTICO (una sola vez, no dentro del circuito)
        # ------------------------------------------------------------
        # edge_table: lista plana de todas las aristas de capa 0 (entrada->oculta),
        # cada una con (columna de dato a leer, wire acumulador donde escribir).
        # zz_table: lista de transferencias de multiplicación (wire_a, wire_b).
        # output_table: lista de aristas oculta->salida ya resueltas a wires.
        edge_table = []       # [{'in_col':.., 'acc_wire':.., 'coefs':[...]}]
        zz_table = []         # [{'wire_a':.., 'wire_b':..}]
        hidden_final_wire = []  # por nodo oculto sobreviviente: wire que representa su valor colapsado
        output_table = []     # [{'src_wire':.., 'coefs':[...]}]  (o marca de "mismo wire de salida")

        for node in graph["hidden_nodes"]:
            raw_carrier_wires = []
            for group in node["edge_groups"]:
                if not group:
                    continue
                # El extractor ya asignó el MISMO wire dedicado a todas las aristas
                # de este grupo (uno por acumulador/neurona-cruda, no por input).
                acc_wire = group[0]["wire"]
                for edge in group:
                    assert edge["wire"] == acc_wire, (
                        "Todas las aristas de un mismo grupo deben compartir wire "
                        "dedicado; revisa la asignación en el extractor."
                    )
                    edge_table.append({
                        "in_col": edge["col"],   # de dónde se LEE el dato clásico (columna filtrada)
                        "acc_wire": acc_wire,    # en qué wire se ESCRIBE la rotación (acumulador)
                        "coefs": edge["coefs"],
                    })
                raw_carrier_wires.append(acc_wire)

            if not raw_carrier_wires:
                continue

            if node["type"] == "mult" and len(raw_carrier_wires) > 1:
                # Cadena de IsingZZ: aproximación estándar para arity > 2
                # (IsingZZ es una interacción de 2 cuerpos; para arity>2 esto
                # es una aproximación por composición en cadena, no un
                # producto N-ario exacto).
                base = raw_carrier_wires[0]
                for other in raw_carrier_wires[1:]:
                    zz_table.append({"wire_a": other, "wire_b": base})
                hidden_final_wire.append(base)
            else:
                hidden_final_wire.append(raw_carrier_wires[0])

        # Wire de salida: el del nodo oculto con más aristas (heurística simple
        # y determinista; cualquier wire activo serviría como acumulador final).
        if hidden_final_wire:
            output_wire = hidden_final_wire[0]
        else:
            output_wire = 0

        for oe in graph["output_edges"]:
            src_wire = hidden_final_wire[oe["hidden_idx"]]
            output_table.append({"src_wire": src_wire, "coefs": oe["coefs"]})

        self._edge_table = edge_table
        self._zz_table = zz_table
        self._output_table = output_table
        self._output_wire = output_wire

        print(f"[QKAN] Plan construido: {len(edge_table)} aristas de entrada, "
              f"{len(zz_table)} transferencias IsingZZ (multiplicación), "
              f"{len(output_table)} aristas de salida. Wire de salida: {output_wire}.")

        # ------------------------------------------------------------
        # Parámetros entrenables (uno por arista/transferencia, forma (degree+1,)
        # para las aristas de re-uploading, escalar para las IsingZZ/salida)
        # ------------------------------------------------------------
        self.edge_weights = nn.Parameter(
            torch.stack([torch.tensor(e["coefs"], dtype=torch.float32) for e in edge_table])
            if edge_table else torch.zeros((0, self.degree + 1))
        )
        self.zz_weights = nn.Parameter(torch.zeros(len(zz_table)))
        self.output_weights = nn.Parameter(
            torch.stack([torch.tensor(o["coefs"], dtype=torch.float32) for o in output_table])
        )

        self.backend_mode = backend_mode
        self.dev = self._initialize_device()
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _initialize_device(self):
        if self.backend_mode == "noisy":
            print("[QKAN] Configurando simulador ruidoso (FakeManilaV2 + NoiseModel)...")
            from qiskit_ibm_runtime.fake_provider import FakeManilaV2
            from qiskit_aer.noise import NoiseModel

            fake_backend = FakeManilaV2()
            noise_model = NoiseModel.from_backend(fake_backend)

            return qml.device(
                "qiskit.aer",
                wires=self.n_qubits,
                backend="aer_simulator_density_matrix",
                noise_model=noise_model,
                shots=1024,
            )
        elif self.backend_mode == "shots":
            print("[QKAN] Configurando simulador por muestreo (shots)...")
            return qml.device("default.qubit", wires=self.n_qubits, shots=1024)
        else:
            print("[QKAN] Configurando simulador ideal (lightning.qubit)...")
            return qml.device("lightning.qubit", wires=self.n_qubits)

    def _qkan_edge(self, x_val, weights, wire):
        """Re-uploading a base de Chebyshev sobre `wire`. Llamar esto varias
        veces sobre el MISMO wire para distintas entradas ACUMULA (suma) sus
        contribuciones, porque las rotaciones RZ sobre el mismo eje componen
        aditivamente sus ángulos."""
        theta = torch.acos(torch.clamp(x_val, -0.9999, 0.9999))
        for i in range(self.degree):
            qml.RY(weights[i], wires=wire)
            qml.RZ(theta, wires=wire)
        qml.RY(weights[self.degree], wires=wire)

    def _circuit(self, inputs):
        # --- Stage 1: entradas -> nodos ocultos (suma gratis + mult vía ZZ) ---
        for idx, edge in enumerate(self._edge_table):
            # Lee el dato clásico de su columna (in_col); escribe la rotación en su
            # wire dedicado (acc_wire). Si el mismo input alimenta varias ramas,
            # aparecerá aquí varias veces con el mismo in_col pero distinto acc_wire.
            self._qkan_edge(inputs[edge["in_col"]], self.edge_weights[idx], wire=edge["acc_wire"])

        for idx, zz in enumerate(self._zz_table):
            qml.IsingZZ(self.zz_weights[idx], wires=[zz["wire_a"], zz["wire_b"]])
            qml.CNOT(wires=[zz["wire_a"], zz["wire_b"]])

        # --- Stage 2: nodos ocultos -> salida (lectura variacional, ver docstring) ---
        for idx, oe in enumerate(self._output_table):
            if oe["src_wire"] == self._output_wire:
                qml.RY(self.output_weights[idx][0], wires=self._output_wire)
            else:
                qml.IsingZZ(self.output_weights[idx][0], wires=[oe["src_wire"], self._output_wire])
                qml.CNOT(wires=[oe["src_wire"], self._output_wire])

        return qml.expval(qml.PauliZ(self._output_wire))

    def forward(self, x):
        x_filtered = x[:, self.active_inputs]
        batch_size = x_filtered.shape[0]
        outputs = torch.zeros(batch_size, device=x.device)
        for i in range(batch_size):
            outputs[i] = self.qnode(x_filtered[i])
        return outputs

    def plot_circuit(self, save_path):
        print(f"[QKAN] Generando diagrama del circuito en {save_path}...")
        # OJO: el argumento del qnode es el vector de DATOS clásicos filtrados
        # (uno por columna en active_inputs), no uno por wire — son cantidades
        # distintas ahora que un input puede escribir en varios wires.
        dummy_inputs = torch.rand(len(self.active_inputs))
        fig, ax = qml.draw_mpl(self.qnode, decimals=2, style="pennylane")(dummy_inputs)
        plt.title(f"QKAN estructurado ({self.n_qubits} qubits, "
                  f"{len(self._zz_table)} nodos-mult vivos)", fontsize=20)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()