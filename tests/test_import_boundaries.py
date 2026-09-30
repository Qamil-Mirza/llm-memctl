"""The dependency rule of the design: who may import whom.

If one of these fails, a module has reached across a boundary that keeps
controllers, environments, agents and analysis swappable.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "memctl"


def imports_of(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return {name for name in found if name.startswith("memctl")}


def offenders(folder: str, forbidden: tuple[str, ...], except_files: tuple[str, ...] = ()) -> list[str]:
    bad = []
    for path in sorted((ROOT / folder).rglob("*.py")):
        if path.name in except_files:
            continue
        for name in imports_of(path):
            if name.startswith(forbidden):
                bad.append(f"{path.relative_to(ROOT)} imports {name}")
    return bad


def test_memory_core_imports_nothing_else_from_memctl():
    allowed = ("memctl.memory",)
    bad = [
        f"{path.name} imports {name}"
        for path in (ROOT / "memory").glob("*.py")
        for name in imports_of(path)
        if not name.startswith(allowed)
    ]
    assert bad == []


def test_controllers_do_not_import_environments_agents_harness_or_analysis():
    forbidden = ("memctl.envs", "memctl.agents", "memctl.harness", "memctl.analysis", "memctl.runlog")
    assert offenders("controllers", forbidden) == []


def test_only_the_oracle_controller_imports_hindsight():
    assert offenders("controllers", ("memctl.hindsight",), except_files=("oracle.py",)) == []


def test_environments_do_not_import_controllers_agents_or_the_harness():
    forbidden = ("memctl.controllers", "memctl.agents", "memctl.harness", "memctl.analysis")
    assert offenders("envs", forbidden) == []


def test_agents_do_not_import_controllers_or_the_harness():
    assert offenders("agents", ("memctl.controllers", "memctl.harness", "memctl.analysis")) == []


def test_analysis_reads_files_and_imports_no_runtime_component():
    forbidden = ("memctl.harness", "memctl.controllers", "memctl.envs", "memctl.agents", "memctl.memory")
    assert offenders("analysis", forbidden) == []
