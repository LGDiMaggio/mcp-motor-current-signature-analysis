"""MCP Server for Motor Current Signature Analysis (MCSA).

Run as a CLI:
    mcp-server-mcsa

Or via Python:
    python -m mcp_server_mcsa
"""

from __future__ import annotations

__version__ = "0.1.0"


def main() -> None:
    """CLI entry point for the MCSA MCP server."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="mcp-server-mcsa",
        description="MCP server for Motor Current Signature Analysis (MCSA)",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    args = parser.parse_args()

    from mcp_server_mcsa.server import serve

    serve(transport=args.transport)


if __name__ == "__main__":
    main()
