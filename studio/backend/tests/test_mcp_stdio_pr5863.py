"""Verification tests for PR #5863 (stdio MCP server support).

Covers the pure helpers, the route-level _validate_url gate, and that the
UNSLOTH_STUDIO_ALLOW_STDIO_MCP gate blocks the stdio transport at every
enforcement point (create/update/test/refresh/discovery/execute) when disabled
and reaches it when enabled. The transport is stubbed so no subprocess spawns;
a recorder asserts whether it was reached.
"""

import os
import sys

import pytest
from fastapi import HTTPException

from core.inference import mcp_client
from storage import mcp_servers_db
from utils import host_policy


def _reset_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", False)
    # The discovered-tool cache is process-global and keyed by server id; tests
    # reuse "stdio1", so clear it (and the failure cool-off) for isolation —
    # otherwise a prior test's warm cache makes discovery skip its probe.
    mcp_client.invalidate_tool_cache()


def _enable(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")


def _disable(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", raising = False)


@pytest.fixture(autouse = True)
def _isolate_stdio_env():
    # apply_stdio_mcp_loopback_default() mutates os.environ and a module flag that
    # monkeypatch can't roll back, and stdio_mcp_enabled() reads the process tool
    # policy; snapshot/restore all three so nothing leaks between tests or files.
    from state import tool_policy

    saved = os.environ.get("UNSLOTH_STUDIO_ALLOW_STDIO_MCP")
    saved_policy = tool_policy.get_tool_policy()
    host_policy._reset_loopback_default_state()
    yield
    host_policy._reset_loopback_default_state()
    tool_policy.set_tool_policy(saved_policy)
    if saved is None:
        os.environ.pop("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", None)
    else:
        os.environ["UNSLOTH_STUDIO_ALLOW_STDIO_MCP"] = saved


# ── transport stub + recorder ───────────────────────────────────────


class _FakeTool:
    def __init__(self, name):
        self._name = name

    def model_dump(self, exclude_none = True):
        return {"name": self._name, "description": f"{self._name} tool"}


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResult:
    is_error = False

    def __init__(self, text):
        self.content = [_Block(text)]


class _RecordingClient:
    """Stand-in for fastmcp.Client; records that the transport was opened."""

    def __init__(self, url, headers, use_oauth, recorder):
        recorder.append({"url": url, "headers": headers, "use_oauth": use_oauth})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def list_tools(self):
        return [_FakeTool("list_directory"), _FakeTool("write_file")]

    async def call_tool(
        self,
        name,
        args,
        raise_on_error = True,
    ):
        return _FakeResult(f"called {name}")


@pytest.fixture
def transport(monkeypatch):
    """Patch mcp_client._client with a recorder. Returns the recorder list;
    empty == stdio transport never reached."""
    recorder = []
    monkeypatch.setattr(
        mcp_client,
        "_client",
        lambda url, headers, use_oauth = False: _RecordingClient(url, headers, use_oauth, recorder),
    )
    return recorder


# ── 1. is_stdio ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "addr",
    [
        "http://localhost:8000/mcp",
        "https://example.com/mcp",
        "  https://example.com/mcp  ",
        "HTTPS://EXAMPLE.COM/mcp",
    ],
)
def test_is_stdio_false_for_http(addr):
    assert mcp_client.is_stdio(addr) is False


@pytest.mark.parametrize(
    "addr",
    [
        "npx -y @modelcontextprotocol/server-filesystem /tmp",
        "python -m some.module",
        "uvx some-server --flag",
        "/usr/local/bin/my-server",
    ],
)
def test_is_stdio_true_for_commands(addr):
    assert mcp_client.is_stdio(addr) is True


# ── 2. parse_stdio_command ──────────────────────────────────────────


def test_parse_basic_argv():
    assert mcp_client.parse_stdio_command(
        "npx -y @modelcontextprotocol/server-filesystem /tmp"
    ) == ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]


def test_parse_keeps_url_argument_as_one_command():
    # gemini "high": a :// inside an ARGUMENT must not break the command.
    assert mcp_client.parse_stdio_command("npx server --endpoint https://example.com/mcp") == [
        "npx",
        "server",
        "--endpoint",
        "https://example.com/mcp",
    ]


def test_parse_quoted_arg():
    assert mcp_client.parse_stdio_command('python -m mod --name "a b"') == [
        "python",
        "-m",
        "mod",
        "--name",
        "a b",
    ]


def test_parse_empty_returns_empty_list():
    assert mcp_client.parse_stdio_command("   ") == []


def test_parse_unclosed_quote_raises_valueerror():
    with pytest.raises(ValueError):
        mcp_client.parse_stdio_command('npx "unclosed')


def test_parse_windows_strips_wrapping_quotes(monkeypatch):
    # gemini "medium": posix=False keeps backslash paths but also the
    # wrapping quotes; the PR strips a matched pair so argv[0] is clean.
    monkeypatch.setattr(sys, "platform", "win32")
    parts = mcp_client.parse_stdio_command(r'"C:\Program Files\node\node.exe" server.js')
    assert parts[0] == r"C:\Program Files\node\node.exe"
    assert parts[1] == "server.js"


# ── 3. stdio_mcp_enabled ────────────────────────────────────────────


@pytest.mark.parametrize("val", ["0", "false", "true", "", " 1 ", "yes", "2"])
def test_stdio_disabled_for_non_exact_one(monkeypatch, val):
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", val)
    assert mcp_client.stdio_mcp_enabled() is False


def test_stdio_enabled_only_for_exact_one(monkeypatch):
    _disable(monkeypatch)
    assert mcp_client.stdio_mcp_enabled() is False
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    assert mcp_client.stdio_mcp_enabled() is True


# ── 3b. loopback bind defaults the gate on ──────────────────────────


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "LOCALHOST", "::1"])
def test_is_external_host_false_for_loopback(host):
    assert host_policy.is_external_host(host) is False


# 127.0.0.2 is loopback in principle, but the rest of the stack hard-codes
# 127.0.0.1, so only the exact aliases count as local here.
@pytest.mark.parametrize("host", ["0.0.0.0", "::", "127.0.0.2", "192.168.1.10", "example.com"])
def test_is_external_host_true_for_network(host):
    assert host_policy.is_external_host(host) is True


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "LOCALHOST", "::1"])
def test_loopback_bind_enables_stdio(monkeypatch, host):
    _disable(monkeypatch)
    host_policy.apply_stdio_mcp_loopback_default(host)
    assert mcp_client.stdio_mcp_enabled() is True


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "127.0.0.2", "192.168.1.10", "example.com"])
def test_network_bind_leaves_stdio_off(monkeypatch, host):
    _disable(monkeypatch)
    host_policy.apply_stdio_mcp_loopback_default(host)
    assert mcp_client.stdio_mcp_enabled() is False


def test_colab_loopback_does_not_auto_enable(monkeypatch):
    # Colab loopback is a hosted VM reachable via the proxy, so it stays off.
    _disable(monkeypatch)
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1", is_colab = True)
    assert mcp_client.stdio_mcp_enabled() is False


def test_explicit_enable_survives_colab(monkeypatch):
    # An explicit operator opt-in still wins over the Colab exclusion (apply_
    # early-returns on an explicit value, before the is_colab check).
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1", is_colab = True)
    assert mcp_client.stdio_mcp_enabled() is True


def test_explicit_disable_survives_loopback(monkeypatch):
    # An explicit =0 must not be overridden by the loopback auto-default.
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "0")
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    assert mcp_client.stdio_mcp_enabled() is False


def test_explicit_enable_survives_network_bind(monkeypatch):
    # A deliberate network opt-in (-H 0.0.0.0 + var=1) must not be clobbered.
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    host_policy.apply_stdio_mcp_loopback_default("0.0.0.0")
    assert mcp_client.stdio_mcp_enabled() is True


def test_loopback_default_not_inherited_by_later_public_bind(monkeypatch):
    # Reusing run_server in one process: a loopback launch auto-enables, a later
    # 0.0.0.0 launch must take it back down (not inherit it as an opt-in).
    _disable(monkeypatch)
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    assert mcp_client.stdio_mcp_enabled() is True
    host_policy.apply_stdio_mcp_loopback_default("0.0.0.0")
    assert mcp_client.stdio_mcp_enabled() is False


def test_remote_access_suspends_only_automatic_stdio_default(monkeypatch):
    _disable(monkeypatch)
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    assert mcp_client.stdio_mcp_enabled() is True
    host_policy.set_remote_connector_active(True)
    assert mcp_client.stdio_mcp_enabled() is False
    host_policy.set_remote_connector_active(False)
    assert mcp_client.stdio_mcp_enabled() is True

    host_policy._reset_loopback_default_state()
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    host_policy.set_remote_connector_active(True)
    assert mcp_client.stdio_mcp_enabled() is True


@pytest.mark.parametrize("second_host", ["127.0.0.1", "0.0.0.0"])
def test_force_disable_after_auto_default_in_same_process(monkeypatch, second_host):
    # Reuse: a loopback launch auto-enables, then the operator sets =0 before a
    # later launch. The force-disable must win whether the later bind is loopback
    # (must not rewrite to 1) or public (the relinquish path must not pop the =0).
    _disable(monkeypatch)
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    assert mcp_client.stdio_mcp_enabled() is True
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "0")
    host_policy.apply_stdio_mcp_loopback_default(second_host)
    assert mcp_client.stdio_mcp_enabled() is False


def test_cleared_env_after_auto_default_falls_back_to_host_default(monkeypatch):
    # Unsetting the var (unlike =0) is "no preference", so a loopback re-apply
    # re-enables -- the asymmetry the staleness guard documents.
    _disable(monkeypatch)
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    monkeypatch.delenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", raising = False)
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    assert mcp_client.stdio_mcp_enabled() is True


def test_disable_tools_overrides_loopback_default(monkeypatch):
    # When stdio is on only via the loopback auto-default, --disable-tools (the
    # only way tool policy is False on a loopback bind) turns it back off.
    from state import tool_policy

    _disable(monkeypatch)
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    assert mcp_client.stdio_mcp_enabled() is True
    tool_policy.set_tool_policy(False)
    assert mcp_client.stdio_mcp_enabled() is False


def test_explicit_env_opt_in_survives_external_default_policy(monkeypatch):
    # `UNSLOTH_STUDIO_ALLOW_STDIO_MCP=1 unsloth studio run -H 0.0.0.0` with no
    # --enable-tools: tool policy is False by the external-host default, not by
    # --disable-tools, so the explicit env opt-in must still win.
    from state import tool_policy

    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    host_policy.apply_stdio_mcp_loopback_default("0.0.0.0")  # no-op: value is explicit
    tool_policy.set_tool_policy(False)
    assert mcp_client.stdio_mcp_enabled() is True


def test_explicit_env_opt_in_beats_disable_tools_on_loopback(monkeypatch):
    # An operator who hand-sets =1 before launch outranks --disable-tools even on
    # loopback: apply_ leaves the auto-default inactive, so the veto doesn't apply.
    from state import tool_policy

    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")  # no-op: value is explicit
    tool_policy.set_tool_policy(False)
    assert mcp_client.stdio_mcp_enabled() is True


@pytest.mark.parametrize("policy", [None, True])
def test_non_false_tool_policy_defers_to_env(monkeypatch, policy):
    # Only an explicit --disable-tools (False) gates stdio; None/True fall through
    # to the env var so the gate keeps its normal meaning.
    from state import tool_policy

    tool_policy.set_tool_policy(policy)
    _disable(monkeypatch)
    assert mcp_client.stdio_mcp_enabled() is False
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    assert mcp_client.stdio_mcp_enabled() is True


# ── 4. probe_timeout ────────────────────────────────────────────────


def test_probe_timeout_matrix():
    assert mcp_client.probe_timeout("https://x/mcp", False) == 8.0
    assert mcp_client.probe_timeout("https://x/mcp", True) == 305.0
    assert mcp_client.probe_timeout("npx server", False) == 60.0
    # oauth wins regardless of address kind (documented behaviour)
    assert mcp_client.probe_timeout("npx server", True) == 305.0


# ── 5. _validate_url gate ───────────────────────────────────────────


def test_validate_url_gate_off_rejects_stdio(monkeypatch):
    _disable(monkeypatch)
    from routes.mcp_servers import _validate_url

    assert _validate_url("https://example.com/mcp") == "https://example.com/mcp"
    # urlparse reads "localhost:8000" scheme as "localhost", so it lands here too.
    for bad in [
        "npx server",
        "python -m mod",
        "ftp://host",
        "example.com",
        "localhost:8000",
        r"C:\node\node.exe server.js",
    ]:
        with pytest.raises(HTTPException) as exc:
            _validate_url(bad)
        assert exc.value.status_code == 400


def test_validate_url_gate_off_message_depends_on_whitespace(monkeypatch):
    # The message names a command only when the value has whitespace, and
    # never says "desktop app only" (self-hosted can opt in via the env var).
    _disable(monkeypatch)
    from routes.mcp_servers import _validate_url

    with pytest.raises(HTTPException) as exc:
        _validate_url("npx -y @modelcontextprotocol/server-filesystem /tmp")
    cmd = exc.value.detail.lower()
    assert "http://" in cmd and "https://" in cmd
    assert "local command" in cmd
    assert "desktop app" not in cmd

    with pytest.raises(HTTPException) as exc:
        _validate_url("example.com")
    lone = exc.value.detail.lower()
    assert "http://" in lone and "https://" in lone
    assert "local command" not in lone


def test_validate_url_gate_on_accepts_stdio(monkeypatch):
    _enable(monkeypatch)
    from routes.mcp_servers import _validate_url

    assert _validate_url("npx -y server /tmp") == "npx -y server /tmp"
    # http still works when stdio is on
    assert _validate_url("https://x/mcp") == "https://x/mcp"
    # url-bearing argument accepted as a command
    assert _validate_url("npx server --url https://x/mcp") == ("npx server --url https://x/mcp")
    # A lone token is ambiguous; accept it as a command rather than
    # guessing it's a URL (no regression for single binaries).
    assert _validate_url("/usr/local/bin/my-mcp-server") == "/usr/local/bin/my-mcp-server"
    assert _validate_url("mcp-server-sqlite") == "mcp-server-sqlite"
    # empty / unparseable still rejected
    for bad in ["   ", '"unclosed']:
        with pytest.raises(HTTPException) as exc:
            _validate_url(bad)
        assert exc.value.status_code == 400


# ── 6. gate enforcement at every spawn path (mocked transport) ──────


def test_create_route_gate(tmp_path, monkeypatch, transport):
    import asyncio

    from models.mcp_servers import McpServerCreate
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    payload = McpServerCreate(display_name = "FS", url = "npx -y server /tmp")

    _disable(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(routes_mcp.create_mcp_server(payload, current_subject = "u", via_api_key = False))
    assert exc.value.status_code == 400

    _enable(monkeypatch)
    resp = asyncio.run(routes_mcp.create_mcp_server(payload, current_subject = "u", via_api_key = False))
    assert resp.url == "npx -y server /tmp"


def test_update_http_to_stdio_blocked_when_off(tmp_path, monkeypatch):
    import asyncio

    from models.mcp_servers import McpServerUpdate
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    _disable(monkeypatch)
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://a/mcp")
    # editing url -> stdio command must 400 (http->stdio bypass closed)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            routes_mcp.update_mcp_server(
                "s1", McpServerUpdate(url = "npx server"), current_subject = "u", via_api_key = False
            )
        )
    assert exc.value.status_code == 400


def test_test_route_gate(tmp_path, monkeypatch, transport):
    import asyncio

    from models.mcp_servers import McpServerTestRequest
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    req = McpServerTestRequest(url = "npx -y server /tmp")

    _disable(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(routes_mcp.test_mcp_server(req, current_subject = "u", via_api_key = False))
    assert exc.value.status_code == 400
    assert transport == []  # transport never opened

    _enable(monkeypatch)
    res = asyncio.run(routes_mcp.test_mcp_server(req, current_subject = "u", via_api_key = False))
    assert res.ok and res.tool_count == 2
    assert len(transport) == 1


def test_refresh_route_gate(tmp_path, monkeypatch, transport):
    import asyncio

    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    # a stdio row, as if carried over from a desktop DB
    mcp_servers_db.create_server(id = "stdio1", display_name = "FS", url = "npx server")

    _disable(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(routes_mcp.refresh_mcp_server_tools("stdio1", current_subject = "u", via_api_key = False))
    assert exc.value.status_code == 400
    assert transport == []

    _enable(monkeypatch)
    res = asyncio.run(routes_mcp.refresh_mcp_server_tools("stdio1", current_subject = "u", via_api_key = False))
    assert res.ok and res.tool_count == 2
    assert len(transport) == 1


def test_discovery_gate(tmp_path, monkeypatch, transport):
    import asyncio

    from core.inference.tools import get_enabled_mcp_tools

    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(id = "stdio1", display_name = "FS", url = "npx server", is_enabled = True)

    _disable(monkeypatch)
    assert asyncio.run(get_enabled_mcp_tools()) == []
    assert transport == []  # filtered out before any probe

    _enable(monkeypatch)
    specs = asyncio.run(get_enabled_mcp_tools())
    assert len(specs) == 2
    assert len(transport) == 1


def test_execute_gate(tmp_path, monkeypatch, transport):
    from core.inference.tools import execute_tool

    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(id = "stdio1", display_name = "FS", url = "npx server", is_enabled = True)

    _disable(monkeypatch)
    out = execute_tool("mcp__stdio1__list_directory", {"path": "/tmp"})
    assert "disabled on this host" in out
    assert transport == []

    _enable(monkeypatch)
    out = execute_tool("mcp__stdio1__list_directory", {"path": "/tmp"})
    assert out == "called list_directory"
    assert len(transport) == 1


# ── 7. env vars ride headers_json as the subprocess env ─────────────


def test_stdio_env_passed_through(tmp_path, monkeypatch, transport):
    from core.inference.tools import execute_tool

    _reset_db(tmp_path, monkeypatch)
    _enable(monkeypatch)
    mcp_servers_db.create_server(
        id = "stdio1",
        display_name = "FS",
        url = "npx server",
        headers_json = '{"API_KEY": "sk-test"}',
        is_enabled = True,
    )
    execute_tool("mcp__stdio1__list_directory", {})
    assert transport[-1]["headers"] == {"API_KEY": "sk-test"}


# ── 8. stdio management requires an interactive UI session ──────────
#
# The loopback bind auto-enables stdio (section 3b), which is the local user's
# own machine and stays that way. What the gate cannot tell apart on its own is
# WHICH credential is calling: get_current_subject accepts a long-lived,
# exportable sk-unsloth API key as readily as the UI's session JWT, and a stdio
# address is a local command. So the command paths take an interactive session,
# while http(s) MCP management stays open to API keys.


def test_create_stdio_rejected_for_api_key(tmp_path, monkeypatch, transport):
    import asyncio

    from models.mcp_servers import McpServerCreate
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    _enable(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            routes_mcp.create_mcp_server(
                McpServerCreate(display_name = "FS", url = "npx -y server /tmp"),
                current_subject = "u",
                via_api_key = True,
            )
        )
    assert exc.value.status_code == 403
    assert mcp_servers_db.list_servers() == []
    assert transport == []


def test_test_route_stdio_rejected_for_api_key_without_spawning(tmp_path, monkeypatch, transport):
    import asyncio

    from models.mcp_servers import McpServerTestRequest
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    _enable(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            routes_mcp.test_mcp_server(
                McpServerTestRequest(url = "/bin/sh -c id"),
                current_subject = "u",
                via_api_key = True,
            )
        )
    assert exc.value.status_code == 403
    # The whole point: the probe never opened the transport.
    assert transport == []


def test_refresh_stdio_rejected_for_api_key(tmp_path, monkeypatch, transport):
    import asyncio

    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    _enable(monkeypatch)
    mcp_servers_db.create_server(id = "stdio1", display_name = "FS", url = "npx server")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            routes_mcp.refresh_mcp_server_tools(
                "stdio1", current_subject = "u", via_api_key = True
            )
        )
    assert exc.value.status_code == 403
    assert transport == []


def test_update_stdio_row_rejected_for_api_key_even_on_rename(tmp_path, monkeypatch):
    import asyncio

    from models.mcp_servers import McpServerUpdate
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    _enable(monkeypatch)
    mcp_servers_db.create_server(id = "stdio1", display_name = "FS", url = "npx server")
    # No url in the payload: the guard must still key off the stored address,
    # else an API key could re-enable the row or rewrite its subprocess env.
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            routes_mcp.update_mcp_server(
                "stdio1",
                McpServerUpdate(display_name = "renamed"),
                current_subject = "u",
                via_api_key = True,
            )
        )
    assert exc.value.status_code == 403
    assert mcp_servers_db.get_server("stdio1")["display_name"] == "FS"


def test_import_stdio_entry_errors_without_failing_the_batch(tmp_path, monkeypatch):
    import asyncio

    from models.mcp_servers import McpServerImportRequest
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    _enable(monkeypatch)
    config = {
        "mcpServers": {
            "remote": {"url": "https://example.com/mcp"},
            "fs": {"command": "npx", "args": ["-y", "server", "/tmp"]},
        }
    }
    result = asyncio.run(
        routes_mcp.import_mcp_servers(
            McpServerImportRequest(config = config), current_subject = "u", via_api_key = True
        )
    )
    assert [row.display_name for row in result.created] == ["remote"]
    assert any("fs" in err and "UI session" in err for err in result.errors)


def test_stdio_disabled_host_still_400_not_403(tmp_path, monkeypatch):
    """Ordering guard: _validate_url runs first, so a 403 never reveals whether
    stdio is enabled on this host."""
    import asyncio

    from models.mcp_servers import McpServerTestRequest
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    _disable(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            routes_mcp.test_mcp_server(
                McpServerTestRequest(url = "npx server"),
                current_subject = "u",
                via_api_key = True,
            )
        )
    assert exc.value.status_code == 400


def test_delete_stdio_allowed_for_api_key(tmp_path, monkeypatch, transport):
    """Documents the carve-out: delete de-privileges, so it stays open."""
    import asyncio

    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    _enable(monkeypatch)
    mcp_servers_db.create_server(id = "stdio1", display_name = "FS", url = "npx server")
    asyncio.run(routes_mcp.delete_mcp_server("stdio1", current_subject = "u"))
    assert mcp_servers_db.get_server("stdio1") is None


def test_http_management_still_open_to_api_keys(tmp_path, monkeypatch, transport):
    """No regression for programmatic callers: an http(s) MCP server is data,
    not code, so every route still accepts an API key for it."""
    import asyncio

    from models.mcp_servers import McpServerCreate, McpServerTestRequest, McpServerUpdate
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    _enable(monkeypatch)

    created = asyncio.run(
        routes_mcp.create_mcp_server(
            McpServerCreate(display_name = "R", url = "https://example.com/mcp"),
            current_subject = "u",
            via_api_key = True,
        )
    )
    assert created.url == "https://example.com/mcp"

    updated = asyncio.run(
        routes_mcp.update_mcp_server(
            created.id,
            McpServerUpdate(display_name = "R2"),
            current_subject = "u",
            via_api_key = True,
        )
    )
    assert updated.display_name == "R2"

    probe = asyncio.run(
        routes_mcp.test_mcp_server(
            McpServerTestRequest(url = "https://example.com/mcp"),
            current_subject = "u",
            via_api_key = True,
        )
    )
    assert probe.ok

    refreshed = asyncio.run(
        routes_mcp.refresh_mcp_server_tools(
            created.id, current_subject = "u", via_api_key = True
        )
    )
    assert refreshed.ok


def test_stdio_audit_log_holds_no_secrets(tmp_path, monkeypatch, transport, capsys):
    """The audit line names the executable and a digest, never argv or env --
    a stdio command routinely carries credentials in both. (structlog renders to
    stdout here, so read capsys rather than caplog.)"""
    import asyncio

    from models.mcp_servers import McpServerCreate
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    _enable(monkeypatch)
    asyncio.run(
        routes_mcp.create_mcp_server(
            McpServerCreate(
                display_name = "FS",
                url = "npx -y server --token sk-secret",
                headers = {"API_KEY": "sk-env-secret"},
            ),
            current_subject = "u",
            via_api_key = False,
        )
    )
    rendered = capsys.readouterr().out
    assert "mcp_servers.stdio_command" in rendered
    assert "sk-secret" not in rendered
    assert "sk-env-secret" not in rendered
    assert "--token" not in rendered
    assert mcp_client.stdio_log_id("npx -y server --token sk-secret") in rendered
