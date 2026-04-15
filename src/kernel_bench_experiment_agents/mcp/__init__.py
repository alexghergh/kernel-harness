"""KernelBench MCP server package.

This package keeps the MCP surface in one dedicated namespace instead of mixing transport,
workspace access, and harness command dispatch into a single top-level module.
"""

SERVER_NAME = "kernelbench"
SERVER_VERSION = "0.2.0"
DEFAULT_PROTOCOL_VERSION = "2025-03-26"
