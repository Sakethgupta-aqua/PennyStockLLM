"""pyproject.toml must describe what the code really needs - the reproducible-install contract.

Measured 2026-10-02: the code imported eight libraries pyproject did not declare (pandas, pdfplumber,
pypdf, requests, beautifulsoup4, sentence-transformers, torch, pydantic) and declared one it never
imported (langchain-community). A fresh `pip install -e .` produced a package that crashed on import;
only this machine's hand-built environment worked. These tests read the real imports with `ast`, so
the declaration cannot drift again without a red test.
"""
import ast
import re
import sys
import tomllib
from functools import lru_cache
from importlib.metadata import packages_distributions
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _third_party_imports(*roots: str) -> set[str]:
    found: set[str] = set()
    for root in roots:
        path = ROOT / root
        for f in ([path] if path.suffix == ".py" else path.rglob("*.py")):
            for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    found.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    found.add(node.module.split(".")[0])
    return {m for m in found if m not in sys.stdlib_module_names and m not in ("pennystock", "__future__")}


def _names(requirements: list[str]) -> set[str]:
    return {re.split(r"[<>=!~\[; ]", r, maxsplit=1)[0].strip().lower().replace("_", "-")
            for r in requirements}


def _project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


@lru_cache(maxsize=1)
def _installed() -> dict[str, list[str]]:
    return packages_distributions()            # module -> distribution names; slow, so read once


def _distributions(module: str) -> set[str]:
    return {d.lower().replace("_", "-") for d in _installed().get(module, [module])}


def test_every_library_the_package_imports_is_a_declared_dependency():
    declared = _names(_project()["dependencies"])
    missing = sorted(m for m in _third_party_imports("pennystock")
                     if not _distributions(m) & declared)
    assert missing == []


def test_every_library_a_script_imports_is_declared_somewhere():
    project = _project()
    declared = _names(project["dependencies"] +
                      [r for extra in project.get("optional-dependencies", {}).values() for r in extra])
    missing = sorted(m for m in _third_party_imports("scripts", "demo.py")
                     if not _distributions(m) & declared)
    assert missing == []


def test_no_runtime_dependency_is_declared_that_nothing_imports():
    used = {d for m in _third_party_imports("pennystock", "scripts", "demo.py") for d in _distributions(m)}
    unused = sorted(_names(_project()["dependencies"]) - used)
    assert unused == []
