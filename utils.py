class PromptUtils:
    def __init__(self, prompt):
        self.prompt = prompt

    def get_node_info(self, node_id: int | str) -> dict:
        node_id = str(node_id)
        return self.prompt[node_id]

    def is_input_connected(self, node_id: int | str, name: str) -> bool:
        node_info = self.get_node_info(node_id)
        input = node_info.get("inputs", {}).get(name)
        # In reality, input will be a 2-elem list of source node ID + slot
        # if it is connected.
        return input is not None

    @property
    def output_map(self):
        if hasattr(self, "_output_map"):
            return self._output_map
        # A map of (src node ID, slot) -> (dest node ID, "input name")
        self._output_map: dict[tuple[str, int], list[tuple[str, str]]] = {}
        for dest_id, dest_info in self.prompt.items():
            for input_name, input in dest_info.get("inputs", {}).items():
                if isinstance(input, list) and len(input) == 2:
                    # This is apparently a link
                    src_id, src_slot = input
                    out_key = (src_id, src_slot)
                    outputs = self._output_map.get(out_key)
                    if outputs is None:
                        self._output_map[out_key] = outputs = []
                    outputs.append((dest_id, input_name))
        return self._output_map

    def is_output_connected(self, node_id: int | str, slot: int) -> bool:
        out_key = (str(node_id), slot)
        return out_key in self.output_map

    # The assumption is that the given output slot can only be connected
    # to nodes of the same type as the one being checked.
    def get_downstream_nodes(self, node_id: int | str, slot: int) -> list[str]:
        downstream: list[str] = []

        to_check = [str(node_id)]
        while to_check:
            check_id = to_check.pop(0)

            # Yes, consider the starting node downstream from itself as well
            downstream.append(check_id)

            outputs = self.output_map.get((check_id, slot))
            if outputs is not None:
                # NB There can be multiple outputs
                out_ids: list[str] = [link_info[0] for link_info in outputs]
                to_check.extend(out_ids)

        return downstream


class WorkflowUtils:
    def __init__(self, extra_pnginfo):
        # Note: API mode does not have extra_pnginfo (it is None)
        self.extra_pnginfo = extra_pnginfo

    @property
    def node_cache(self) -> dict[int, dict]:
        if hasattr(self, "_node_cache"):
            return self._node_cache
        # TODO Is it worth caching every node? Maybe filter by type?
        self._node_cache: dict[int, dict] = {
            node_info["id"]: node_info
            for node_info in self.extra_pnginfo["workflow"]["nodes"]
        }
        return self._node_cache

    @property
    def link_cache(self) -> dict[int, tuple]:
        if hasattr(self, "_link_cache"):
            return self._link_cache
        self._link_cache: dict[int, tuple] = {
            link_info[0]: link_info
            for link_info in self.extra_pnginfo["workflow"]["links"]
        }
        return self._link_cache

    def get_node_info(self, node_id: int | str) -> dict:
        node_id = int(node_id)
        return self.node_cache[node_id]

    def get_node_info_nocache(self, node_id: int | str) -> dict:
        node_id = int(node_id)
        for node_info in self.extra_pnginfo["workflow"]["nodes"]:
            if node_info["id"] == node_id:
                return node_info
        raise ValueError(f"Node missing from workflow: {node_id}")

    def is_output_connected_nocache(
        self, node_info: dict, type: str | None = None, name: str | None = None
    ) -> bool:
        assert type is not None or name is not None
        outputs = node_info.get("outputs", [])
        return any(
            [
                (
                    (name is not None and output["name"] == name)
                    or (type is not None and output["type"] == type)
                )
                and output["links"]  # Can be an empty list as well
                for output in outputs
            ]
        )

    def is_output_connected(
        self, node_id: int | str, type: str | None = None, name: str | None = None
    ) -> bool:
        node_info = self.get_node_info(node_id)
        return self.is_output_connected_nocache(node_info, type=type, name=name)

    def is_input_connected_nocache(
        self, node_info: dict, type: str | None = None, name: str | None = None
    ) -> bool:
        assert type is not None or name is not None
        inputs = node_info.get("inputs", [])
        return any(
            [
                (
                    (name is not None and input["name"] == name)
                    or (type is not None and input["type"] == type)
                )
                and input["link"] is not None
                for input in inputs
            ]
        )

    def is_input_connected(
        self, node_id: int | str, type: str | None = None, name: str | None = None
    ) -> bool:
        node_info = self.get_node_info(node_id)
        return self.is_input_connected_nocache(node_info, type=type, name=name)

    def get_downstream_nodes(self, node_id: int | str, bus_type: str) -> list[int]:
        downstream: list[int] = []

        to_check = [int(node_id)]
        while to_check:
            check_id = to_check.pop(0)

            # Yes, consider the starting node downstream from itself as well
            downstream.append(check_id)

            node_info = self.get_node_info(check_id)

            outputs = node_info.get("outputs", [])
            bus_out = [out["links"] for out in outputs if out["type"] == bus_type][0]
            if bus_out:
                # NB There can be multiple outputs
                out_ids: list[int] = [
                    self.link_cache[link_id][3]  # dest node id
                    for link_id in bus_out
                ]
                to_check.extend(out_ids)

        return downstream


# The following convenience functions use the "nocache" versions, meant for
# one-off queries.


def is_input_connected(
    extra_pnginfo, node_id: int | str, type: str | None = None, name: str | None = None
) -> bool:
    inst = WorkflowUtils(extra_pnginfo)
    node_info = inst.get_node_info_nocache(node_id)
    return inst.is_input_connected_nocache(node_info, type=type, name=name)


def is_output_connected(
    extra_pnginfo, node_id: int | str, type: str | None = None, name: str | None = None
) -> bool:
    inst = WorkflowUtils(extra_pnginfo)
    node_info = inst.get_node_info_nocache(node_id)
    return inst.is_output_connected_nocache(node_info, type=type, name=name)
