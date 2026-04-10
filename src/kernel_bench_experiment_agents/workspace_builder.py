"""Compatibility entrypoint that re-exports workspace preparation.

Scripts import this name so the CLI surface can stay stable even if workspace preparation lives in another module.
"""

from __future__ import annotations

from .workspace_prepare import command_prepare_problem_workspace

__all__ = [
    "command_prepare_problem_workspace",
]
