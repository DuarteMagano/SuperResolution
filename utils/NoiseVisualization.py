import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


class BackendNoiseVisualizer:
    def __init__(self, backend, gate_names=None, figsize=(20, 18)):
        self.backend = backend
        self.figsize = figsize
        self.gate_names = gate_names or ["rz", "sx", "x", "measure", "ecr", "cx", "cz"]

        self.qubit_data = self._collect_qubit_noise()
        self.gate_data = self._collect_gate_data()

    @staticmethod
    def _safe_float(x):
        return None if x is None else float(x)

    def _qubit_property(self, props, q, key):
        try:
            d = props.qubit_property(q)
            value = d.get(key, (None,))[0]
            return self._safe_float(value)
        except Exception:
            return None

    def _collect_qubit_noise(self):
        props = self.backend.properties()
        n = self.backend.num_qubits

        qubits = []
        t1_us = []
        t2_us = []
        readout_error = []

        for q in range(n):
            qubits.append(q)

            t1 = self._qubit_property(props, q, "T1")
            t2 = self._qubit_property(props, q, "T2")
            ro = self._qubit_property(props, q, "readout_error")

            t1_us.append(None if t1 is None else 1e6 * t1)
            t2_us.append(None if t2 is None else 1e6 * t2)
            readout_error.append(ro)

        return {
            "qubits": np.array(qubits),
            "t1_us": np.array(t1_us, dtype=object),
            "t2_us": np.array(t2_us, dtype=object),
            "readout_error": np.array(readout_error, dtype=object),
        }

    def _collect_gate_data(self):
        target = self.backend.target
        out = {}

        for gate in self.gate_names:
            if gate not in target:
                continue

            qargs_list = []
            errors = []
            durations_ns = []

            for qargs, inst_props in target[gate].items():
                if inst_props is None:
                    continue

                err = getattr(inst_props, "error", None)
                dur = getattr(inst_props, "duration", None)

                qargs_list.append(qargs)
                errors.append(None if err is None else float(err))
                durations_ns.append(None if dur is None else 1e9 * float(dur))

            out[gate] = {
                "qargs": qargs_list,
                "error": np.array(errors, dtype=object),
                "duration_ns": np.array(durations_ns, dtype=object),
            }

        return out

    @staticmethod
    def _valid_xy(x, y):
        x_out, y_out = [], []
        for xi, yi in zip(x, y):
            if yi is not None:
                x_out.append(xi)
                y_out.append(float(yi))
        return np.array(x_out), np.array(y_out)

    @staticmethod
    def _positive_xy(x, y):
        x_out, y_out = [], []
        for xi, yi in zip(x, y):
            if yi is not None and float(yi) > 0:
                x_out.append(xi)
                y_out.append(float(yi))
        return np.array(x_out), np.array(y_out)

    @staticmethod
    def _filter_single_qubit_gate(gate_block):
        x, err, dur = [], [], []

        for qargs, e, d in zip(gate_block["qargs"], gate_block["error"], gate_block["duration_ns"]):
            if len(qargs) != 1:
                continue
            x.append(qargs[0])
            err.append(e)
            dur.append(d)

        return np.array(x), np.array(err, dtype=object), np.array(dur, dtype=object)

    @staticmethod
    def _filter_two_qubit_gate(gate_block):
        labels, err, dur = [], [], []

        for qargs, e, d in zip(gate_block["qargs"], gate_block["error"], gate_block["duration_ns"]):
            if len(qargs) != 2:
                continue
            labels.append(f"{qargs[0]}-{qargs[1]}")
            err.append(e)
            dur.append(d)

        return labels, np.array(err, dtype=object), np.array(dur, dtype=object)

    def _plot_single_qubit_metric(self, ax, x, y, title, ylabel, logy=False):
        if logy:
            xx, yy = self._positive_xy(x, y)
        else:
            xx, yy = self._valid_xy(x, y)

        ax.plot(xx, yy, ".", markersize=5)
        ax.set_title(title)
        ax.set_xlabel("Qubit index")
        ax.set_ylabel(ylabel)
        if logy and len(yy) > 0:
            ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")

    def _plot_multi_single_qubit_gates(self, ax, quantity_key, title, ylabel, gate_order, logy=False):
        for gate in gate_order:
            if gate not in self.gate_data:
                continue

            q, err, dur = self._filter_single_qubit_gate(self.gate_data[gate])
            arr = err if quantity_key == "error" else dur

            if logy:
                xx, yy = self._positive_xy(q, arr)
            else:
                xx, yy = self._valid_xy(q, arr)

            if len(xx) > 0:
                ax.plot(xx, yy, ".", markersize=4, label=gate)

        ax.set_title(title)
        ax.set_xlabel("Qubit index")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend()

    def _plot_two_qubit_sorted(self, ax, quantity_key, title, ylabel, gate_order, logy=False):
        vals = []
        gate_tags = []

        for gate in gate_order:
            if gate not in self.gate_data:
                continue

            _, err, dur = self._filter_two_qubit_gate(self.gate_data[gate])
            arr = err if quantity_key == "error" else dur

            for v in arr:
                if v is None:
                    continue
                v = float(v)
                if logy and v <= 0:
                    continue
                vals.append(v)
                gate_tags.append(gate)

        if len(vals) == 0:
            ax.set_title(title)
            ax.set_xlabel("Couplers sorted by value")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3, which="both")
            return

        vals = np.array(vals)
        gate_tags = np.array(gate_tags)

        idx = np.argsort(vals)
        vals = vals[idx]
        gate_tags = gate_tags[idx]

        for gate in gate_order:
            mask = gate_tags == gate
            if np.any(mask):
                x = np.arange(len(vals))[mask]
                y = vals[mask]
                ax.plot(x, y, ".", markersize=4, label=gate)

        ax.set_title(title)
        ax.set_xlabel("Couplers sorted by value")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend()

    def _plot_top_two_qubit(self, ax, quantity_key, title, ylabel, gate_order, top_k=20, logy=False):
        labels = []
        vals = []
        gate_tags = []

        for gate in gate_order:
            if gate not in self.gate_data:
                continue

            labs, err, dur = self._filter_two_qubit_gate(self.gate_data[gate])
            arr = err if quantity_key == "error" else dur

            for lab, v in zip(labs, arr):
                if v is None:
                    continue
                v = float(v)
                if logy and v <= 0:
                    continue
                labels.append(lab)
                vals.append(v)
                gate_tags.append(gate)

        if len(vals) == 0:
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3, which="both")
            return

        labels = np.array(labels)
        vals = np.array(vals)
        gate_tags = np.array(gate_tags)

        idx = np.argsort(vals)[-top_k:]
        idx = idx[np.argsort(vals[idx])]

        labels = labels[idx]
        vals = vals[idx]
        gate_tags = gate_tags[idx]

        x = np.arange(len(vals))

        for gate in gate_order:
            mask = gate_tags == gate
            if np.any(mask):
                ax.plot(x[mask], vals[mask], ".", markersize=6, label=gate)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend()


    def print_summary(self):
        import numpy as np
        import pandas as pd
        from IPython.display import display

        def stats(arr):
            vals = np.array([float(v) for v in arr if v is not None], dtype=float)
            if len(vals) == 0:
                return None
            return {
                "Min": vals.min(),
                "Mean": vals.mean(),
                "Median": np.median(vals),
                "Max": vals.max(),
            }

        def fmt_value(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "—"
            if v == 0:
                return "0"
            if abs(v) < 1e-2:
                return f"{v:.2e}"
            if abs(v) < 10:
                return f"{v:.3f}"
            if abs(v) < 100:
                return f"{v:.2f}"
            return f"{v:.1f}"

        print(f"Backend: {self.backend.name}")
        print(f"Qubits: {self.backend.num_qubits}")
        print()

        # -------------------------
        # QUBIT TABLE
        # -------------------------
        qubit_rows = []

        for name, arr in [
            ("T1 (µs)", self.qubit_data["t1_us"]),
            ("T2 (µs)", self.qubit_data["t2_us"]),
            ("Readout error", self.qubit_data["readout_error"]),
        ]:
            s = stats(arr)

            if s is None:
                qubit_rows.append(
                    {"Metric": name, "Min": "—", "Mean": "—", "Median": "—", "Max": "—"}
                )
            else:
                qubit_rows.append(
                    {
                        "Metric": name,
                        "Min": fmt_value(s["Min"]),
                        "Mean": fmt_value(s["Mean"]),
                        "Median": fmt_value(s["Median"]),
                        "Max": fmt_value(s["Max"]),
                    }
                )

        qubit_df = pd.DataFrame(qubit_rows)

        # -------------------------
        # GATE TABLE
        # -------------------------
        gate_rows = []

        for gate, block in self.gate_data.items():
            errs = [v for v in block["error"] if v is not None]
            durs = [v for v in block["duration_ns"] if v is not None]

            if len(errs) > 0:
                s = stats(errs)
                gate_rows.append(
                    {
                        "Gate": gate,
                        "Metric": "error",
                        "Min": fmt_value(s["Min"]),
                        "Mean": fmt_value(s["Mean"]),
                        "Median": fmt_value(s["Median"]),
                        "Max": fmt_value(s["Max"]),
                    }
                )

            if len(durs) > 0:
                s = stats(durs)
                gate_rows.append(
                    {
                        "Gate": gate,
                        "Metric": "duration (ns)",
                        "Min": fmt_value(s["Min"]),
                        "Mean": fmt_value(s["Mean"]),
                        "Median": fmt_value(s["Median"]),
                        "Max": fmt_value(s["Max"]),
                    }
                )

        gate_df = pd.DataFrame(gate_rows)

        display(qubit_df.style.hide(axis="index").set_caption("Qubit properties"))
        display(gate_df.style.hide(axis="index").set_caption("Gate properties"))

    def plot_dashboard(self, top_k_couplers=20):
        fig, axes = plt.subplots(5, 2, figsize=self.figsize)
        fig.suptitle(f"Noise characterization — {self.backend.name}", fontsize=18)

        # Row 1: no log scale
        self._plot_single_qubit_metric(
            axes[0, 0],
            self.qubit_data["qubits"],
            self.qubit_data["t1_us"],
            "T1 vs qubit",
            "T1 (µs)",
            logy=False
        )

        self._plot_single_qubit_metric(
            axes[0, 1],
            self.qubit_data["qubits"],
            self.qubit_data["t2_us"],
            "T2 vs qubit",
            "T2 (µs)",
            logy=False
        )

        # Row 2
        self._plot_single_qubit_metric(
            axes[1, 0],
            self.qubit_data["qubits"],
            self.qubit_data["readout_error"],
            "Readout error vs qubit",
            "Readout error",
            logy=True
        )

        self._plot_multi_single_qubit_gates(
            axes[1, 1],
            quantity_key="error",
            title="Single-qubit gate error vs qubit",
            ylabel="Gate error",
            gate_order=["rz", "sx", "x", "measure"],
            logy=True
        )

        # Row 3
        self._plot_two_qubit_sorted(
            axes[2, 0],
            quantity_key="error",
            title="Two-qubit gate error (sorted couplers)",
            ylabel="Gate error",
            gate_order=["ecr", "cx", "cz"],
            logy=True
        )

        self._plot_multi_single_qubit_gates(
            axes[2, 1],
            quantity_key="duration",
            title="Single-qubit gate duration vs qubit",
            ylabel="Duration (ns)",
            gate_order=["rz", "sx", "x", "measure"],
            logy=True
        )

        # Row 4
        self._plot_two_qubit_sorted(
            axes[3, 0],
            quantity_key="duration",
            title="Two-qubit gate duration (sorted couplers)",
            ylabel="Duration (ns)",
            gate_order=["ecr", "cx", "cz"],
            logy=True
        )

        ax = axes[3, 1]
        ro_vals = np.array(
            [np.nan if v is None else float(v) for v in self.qubit_data["readout_error"]],
            dtype=float
        )
        im = ax.imshow(ro_vals.reshape(1, -1), aspect="auto")
        ax.set_title("Readout error heatmap")
        ax.set_xlabel("Qubit index")
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, label="Readout error")

        # Row 5
        self._plot_top_two_qubit(
            axes[4, 0],
            quantity_key="error",
            title=f"Worst {top_k_couplers} two-qubit couplers by error",
            ylabel="Gate error",
            gate_order=["ecr", "cx", "cz"],
            top_k=top_k_couplers,
            logy=True
        )

        self._plot_top_two_qubit(
            axes[4, 1],
            quantity_key="duration",
            title=f"Top {top_k_couplers} longest two-qubit couplers",
            ylabel="Duration (ns)",
            gate_order=["ecr", "cx", "cz"],
            top_k=top_k_couplers,
            logy=True
        )

        plt.tight_layout()
        plt.show()