"""
Tutorial notebooks under tests/ are authored as MyST Markdown (tests/*.md)
and paired with jupytext (see [tool.jupytext] in pyproject.toml). The .md is
the reviewed, git-tracked source; the .ipynb nbval actually executes is a
generated artifact (gitignored) kept in sync here before collection.

This keeps notebook diffs readable in code review while still letting
nbval collect and run real .ipynb files as before.
"""

from pathlib import Path

import jupytext

TESTS_DIR = Path(__file__).parent


def pytest_configure(config):
    for md_path in sorted(TESTS_DIR.glob("*.md")):
        notebook = jupytext.read(md_path)
        ipynb_path = md_path.with_suffix(".ipynb")
        jupytext.write(notebook, ipynb_path)
