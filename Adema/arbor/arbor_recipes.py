import arbor as arb
import os
import json


class optimized_recipe(arb.recipe):
    def __init__(
        self,
        connectome_dir,
        record_soma=True,
        nodes_to_record=None,
        iclamp_schedule=None,
        neurite_params=None,
        cablet="hh",
        cv_max_extent=5.0,
    ):
        super().__init__()

        self.neurite_params = neurite_params
        self.iclamp_schedule = {} if iclamp_schedule is None else iclamp_schedule
        self.nodes_to_record = [] if nodes_to_record is None else nodes_to_record
        self.record_soma = record_soma
        self.connectome_dir = connectome_dir
        self.cablet = cablet
        self.cv_max_extent = cv_max_extent

        with open(os.path.join(self.connectome_dir, "gid_mapping.json"), "r") as file:
            self.mapping = json.load(file)

        self.gid_to_neuron_id = {int(v): int(k) for k, v in self.mapping.items()}
        self.ncells = len(self.mapping)

    def num_cells(self):
        return self.ncells

    def cell_kind(self, gid):
        return arb.cell_kind.cable

    def cell_description(self, gid):
        gp = os.path.join(self.connectome_dir, str(gid))
        pd_path = os.path.join(gp, "decor.arbc")
        pt_path = os.path.join(gp, "morphology.arbc")
        pm_path = os.path.join(gp, "mapping.json")

        morphology = arb.load_component(pt_path).component
        decor = arb.load_component(pd_path).component

        with open(pm_path, "r") as file:
            node_to_segment = json.load(file)

        neuron_id = self.gid_to_neuron_id[gid]

        if self.neurite_params is not None and neuron_id in self.neurite_params.index:
            params = self.neurite_params.loc[neuron_id]

            if "cm" in params.index:
                decor.set_property(cm=float(params["cm"]))
            if "Ra" in params.index:
                decor.set_property(rL=float(params["Ra"]))

            hh_kwargs = {}
            if "dend_gnabar_hh" in params.index:
                hh_kwargs["gnabar"] = float(params["dend_gnabar_hh"])
            if "dend_gkbar_hh" in params.index:
                hh_kwargs["gkbar"] = float(params["dend_gkbar_hh"])
            if "dend_gl_hh" in params.index:
                hh_kwargs["gl"] = float(params["dend_gl_hh"])
            if "el_hh" in params.index:
                hh_kwargs["el"] = float(params["el_hh"])

            decor.paint("(all)", arb.density(self.cablet, **hh_kwargs))
        else:
            decor.paint("(all)", arb.density(self.cablet))

        if neuron_id in self.iclamp_schedule:
            for k, v in self.iclamp_schedule[neuron_id].items():
                if k == "soma":
                    q = 0
                else:
                    q = node_to_segment[k]

                decor.place(
                    f"(on-components 0.5 (segment {q}))",
                    v,
                    f"ic_{k}",
                )

        cv_policy = arb.cv_policy_max_extent(self.cv_max_extent)
        return arb.cable_cell(morphology, decor, None, cv_policy)

    def connections_on(self, gid):
        gp = os.path.join(self.connectome_dir, str(gid))
        pc = os.path.join(gp, "connectors.json")

        with open(pc, "r") as f:
            connections = json.load(f)

        return [
            arb.connection(
                source=tuple(c["source"]),
                dest=c["target"],
                weight=c["weight"],
                delay=c["delay"] * arb.units.ms,
            )
            for c in connections
        ]

    def global_properties(self, kind):
        return arb.neuron_cable_properties()

    def probes(self, gid):
        if not self.record_soma:
            return []
        return [
            arb.cable_probe_membrane_voltage(
                "(on-components 0.5 (segment 0))",
                "soma",
            )
        ]
