"""Small standard-library client for ComfyUI API-format workflows."""

from __future__ import annotations

import copy
import json
import time
import uuid
from pathlib import Path
from urllib import error, request


class WorkflowError(ValueError):
    """Raised when a workflow is not safe to queue."""


def read_workflow(path: Path) -> dict:
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"{path}: cannot read JSON: {exc}") from exc
    errors = validate_workflow(graph)
    if errors:
        raise WorkflowError(f"{path}: " + "; ".join(errors))
    return graph


def validate_workflow(graph: object) -> list[str]:
    """Validate the parts of API format that do not need a live node schema."""
    if not isinstance(graph, dict):
        return ["top level must be an object"]
    if "nodes" in graph:
        return ["UI-format workflow; export or save it in API format"]

    errors: list[str] = []
    for node_id, node in graph.items():
        if not isinstance(node_id, str) or not isinstance(node, dict):
            errors.append(f"node {node_id!r} is not an object")
            continue
        if not isinstance(node.get("class_type"), str):
            errors.append(f"node {node_id} has no class_type")
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            errors.append(f"node {node_id} has no inputs object")
            continue
        for input_name, value in inputs.items():
            if not is_link(value):
                continue
            source_id, output_slot = value
            if source_id not in graph:
                errors.append(
                    f"node {node_id}.{input_name} references missing node {source_id}"
                )
            if output_slot < 0:
                errors.append(
                    f"node {node_id}.{input_name} has negative output slot {output_slot}"
                )
    return errors


def is_link(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def parse_value(raw: str) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_overrides(graph: dict, overrides: list[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise WorkflowError(f"override must be NODE_ID.INPUT=value: {override}")
        key, raw_value = override.split("=", 1)
        if "." not in key:
            raise WorkflowError(f"override must name NODE_ID.INPUT: {key}")
        node_id, input_name = key.split(".", 1)
        if node_id not in graph:
            raise WorkflowError(f"override references missing node {node_id}")
        inputs = graph[node_id].get("inputs", {})
        if input_name not in inputs:
            raise WorkflowError(f"node {node_id} has no input {input_name!r}")
        inputs[input_name] = parse_value(raw_value)


def find_nodes(graph: dict, class_type: str) -> list[str]:
    return [node_id for node_id, node in graph.items() if node["class_type"] == class_type]


def with_output_suffix(graph: dict, suffix: str) -> dict:
    """Clone a graph and separate its saved files from another test arm."""
    updated = copy.deepcopy(graph)
    changed = 0
    for node in updated.values():
        inputs = node["inputs"]
        prefix = inputs.get("filename_prefix")
        if isinstance(prefix, str):
            inputs["filename_prefix"] = f"{prefix}_{suffix}"
            changed += 1
    if not changed:
        raise WorkflowError("workflow has no filename_prefix outputs to isolate")
    return updated


def output_refs(history: dict) -> list[str]:
    refs: list[str] = []
    for node_outputs in history.get("outputs", {}).values():
        if not isinstance(node_outputs, dict):
            continue
        for items in node_outputs.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and item.get("filename"):
                    subfolder = str(item.get("subfolder", "")).strip("/")
                    name = str(item["filename"])
                    refs.append(f"{subfolder}/{name}" if subfolder else name)
    return refs


def history_error(history: dict) -> str | None:
    status = history.get("status", {})
    if status.get("status_str") == "error":
        messages = status.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, list) and len(message) == 2:
                detail = message[1]
                if isinstance(detail, dict):
                    return str(detail.get("exception_message") or detail)
        return "ComfyUI reported an execution error"
    return None


class ComfyClient:
    def __init__(self, server: str, request_timeout: int = 30):
        self.server = server.rstrip("/")
        self.request_timeout = request_timeout

    def _json(self, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = request.Request(
            f"{self.server}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=self.request_timeout) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ComfyUI HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"cannot reach {self.server}: {exc.reason}") from exc
        return json.loads(body) if body else {}

    def check(self) -> None:
        self._json("/system_stats")

    def free_models(self) -> None:
        self._json("/free", {"unload_models": True, "free_memory": True})

    def queue(self, graph: dict) -> str:
        response = self._json(
            "/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())}
        )
        prompt_id = response.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return prompt_id: {response}")
        return str(prompt_id)

    def wait(self, prompt_id: str, timeout: int, poll_seconds: float = 2.0) -> dict:
        started = time.monotonic()
        while time.monotonic() - started <= timeout:
            history = self._json(f"/history/{prompt_id}")
            if prompt_id in history:
                return history[prompt_id]
            time.sleep(poll_seconds)
        raise TimeoutError(f"timed out after {timeout}s waiting for {prompt_id}")
