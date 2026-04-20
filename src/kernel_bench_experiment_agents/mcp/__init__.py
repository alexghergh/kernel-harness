"""KernelBench MCP server package.

The harness-specific MCP layer only defines context, resources, tool handlers, and synthetic trace
logging. Transport, protocol lifecycle, and initialization now come from the official Python MCP SDK.
"""

SERVER_NAME = "kernelbench"
SERVER_VERSION = "0.4.0"
