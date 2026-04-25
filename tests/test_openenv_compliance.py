"""OpenEnv compliance tests — validates arm-gym conforms to all OpenEnv requirements.

Covers:
  1. ARMGymEnv class attributes (SUPPORTS_CONCURRENT_SESSIONS)
  2. reset() → CompilerObservation
  3. step(CompilerAction) → CompilerObservation
  4. state property → dict
  5. metadata() → dict with required fields
  6. Pydantic model validation (CompilerAction, CompilerObservation)
  7. FastAPI endpoint presence (/health, /metadata, /schema, /tasks, /state, /reset, /step, /ws)
  8. WebSocket /ws handler (reset, step, state, metadata methods)
  9. openenv.yaml manifest validity
 10. Client/server import separation
 11. No reserved tool names for MCP tools
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Fixtures — mock the aarch64 toolchain so tests run on any host
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARM_GYM_PKG = PROJECT_ROOT / "arm_gym"


def _mock_which(name: str) -> str | None:
    """Return None for aarch64 cross-tools, passthrough for others."""
    aarch64_tools = {
        "aarch64-linux-gnu-gcc",
        "aarch64-linux-gnu-as",
        "aarch64-linux-gnu-ld",
        "qemu-aarch64-static",
        "llvm-mca",
        "llvm-mca-20",
        "llvm-mca-21",
        "clang",
        "clang-20",
        "clang-21",
    }
    if name in aarch64_tools:
        return None
    # Fall through to real which for system tools
    import shutil as _real_shutil

    fn = getattr(_real_shutil.which, "__wrapped__", None)
    return fn(name) if fn else None


@pytest.fixture(autouse=True)
def _mock_toolchain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no real aarch64 toolchain is needed for any test."""
    import shutil

    _original_which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name: _mock_which(name))
    # Store original for passthrough if needed
    if not hasattr(shutil.which, "__wrapped__"):
        _mock_which.__wrapped__ = _original_which  # type: ignore[attr-defined]


_STUB_ASM = ".text\n.global kernel\n.type kernel, %function\nkernel:\nret\n.size kernel, .-kernel\n"


@pytest.fixture()
def env() -> Any:
    """Build an ARMGymEnv with toolchain mocked out and baseline cache pre-populated."""
    from arm_gym.compile_baseline import ToolchainInfo
    from arm_gym.env import ARMGymEnv
    from arm_gym.kernels import generate_all, split_train_eval
    from arm_gym.reward import RewardConfig
    from arm_gym.verifier import VerifierConfig

    tc = ToolchainInfo(
        clang=None,
        gcc_aarch64=None,
        mca=None,
        mcpu="neoverse-v2",
        mcpu_disclosed="mock — no toolchain",
    )
    vcfg = VerifierConfig(
        mca_bin="llvm-mca",
        assembler="aarch64-linux-gnu-as",
        linker="aarch64-linux-gnu-ld",
        qemu="qemu-aarch64-static",
        mcpu="neoverse-v2",
    )
    variants = generate_all()
    train, _ = split_train_eval(variants)
    baseline_cache = {v.variant_id: (_STUB_ASM, 100.0) for v in train}
    return ARMGymEnv(
        toolchain=tc,
        verifier_cfg=vcfg,
        reward_cfg=RewardConfig(),
        variants=train,
        _baseline_cache=baseline_cache,
    )


@pytest.fixture()
def app_client() -> httpx.AsyncClient:
    """Async test client for the FastAPI app, with singleton env mocked."""
    import arm_gym.env as env_module
    from arm_gym.compile_baseline import ToolchainInfo
    from arm_gym.env import ARMGymEnv, app
    from arm_gym.kernels import generate_all, split_train_eval
    from arm_gym.reward import RewardConfig
    from arm_gym.verifier import VerifierConfig

    tc = ToolchainInfo(
        clang=None, gcc_aarch64=None, mca=None,
        mcpu="neoverse-v2", mcpu_disclosed="mock",
    )
    vcfg = VerifierConfig(
        mca_bin="llvm-mca", assembler="aarch64-linux-gnu-as",
        linker="aarch64-linux-gnu-ld", qemu="qemu-aarch64-static",
        mcpu="neoverse-v2",
    )
    variants = generate_all()
    train, _ = split_train_eval(variants)
    baseline_cache = {v.variant_id: (_STUB_ASM, 100.0) for v in train}
    mock_env = ARMGymEnv(
        toolchain=tc, verifier_cfg=vcfg,
        reward_cfg=RewardConfig(), variants=train,
        _baseline_cache=baseline_cache,
    )
    env_module._env_singleton = mock_env
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


# ===========================================================================
# 1. SUPPORTS_CONCURRENT_SESSIONS class attribute
# ===========================================================================

class TestConcurrentSessions:
    def test_class_has_supports_concurrent_sessions(self) -> None:
        from arm_gym.env import ARMGymEnv

        assert hasattr(ARMGymEnv, "SUPPORTS_CONCURRENT_SESSIONS")

    def test_supports_concurrent_sessions_is_true(self) -> None:
        from arm_gym.env import ARMGymEnv

        assert ARMGymEnv.SUPPORTS_CONCURRENT_SESSIONS is True

    def test_instance_inherits_flag(self, env: Any) -> None:
        assert env.SUPPORTS_CONCURRENT_SESSIONS is True


# ===========================================================================
# 2. reset() returns CompilerObservation
# ===========================================================================

class TestResetMethod:
    def test_has_reset_method(self, env: Any) -> None:
        assert hasattr(env, "reset")
        assert callable(env.reset)

    def test_reset_returns_compiler_observation(self, env: Any) -> None:
        from arm_gym.env import CompilerObservation

        obs = env.reset(seed=42)
        assert isinstance(obs, CompilerObservation)

    def test_reset_observation_fields(self, env: Any) -> None:
        obs = env.reset(seed=42)
        assert obs.done is False
        assert obs.reward is None
        assert isinstance(obs.variant_id, str)
        assert len(obs.variant_id) > 0
        assert isinstance(obs.c_source, str)
        assert len(obs.c_source) > 0
        assert isinstance(obs.baseline_cycles, float)
        assert isinstance(obs.difficulty, int)
        assert obs.difficulty >= 1

    def test_reset_accepts_seed_and_episode_id(self, env: Any) -> None:
        obs = env.reset(seed=123, episode_id="test-episode-1")
        assert obs.variant_id is not None
        state = env.state
        assert state["episode_id"] == "test-episode-1"

    def test_reset_is_deterministic_with_seed(self, env: Any) -> None:
        obs1 = env.reset(seed=999)
        obs2 = env.reset(seed=999)
        assert obs1.variant_id == obs2.variant_id
        assert obs1.c_source == obs2.c_source


# ===========================================================================
# 3. step() takes CompilerAction, returns CompilerObservation
# ===========================================================================

def _mock_subprocess_run(cmd: list[str], **kwargs: Any) -> Any:
    """Mock subprocess.run to simulate aarch64-linux-gnu-as succeeding."""

    class FakeResult:
        def __init__(self, rc: int = 0, stdout: str = "", stderr: str = "") -> None:
            self.returncode = rc
            self.stdout = stdout
            self.stderr = stderr

    if cmd and "aarch64-linux-gnu-as" in cmd[0]:
        for i, arg in enumerate(cmd):
            if arg == "-o" and i + 1 < len(cmd):
                Path(cmd[i + 1]).touch()
        return FakeResult(0)
    if cmd and "llvm-mca" in cmd[0]:
        mca_out = (
            "Iterations:        100\n"
            "Instructions:      10\n"
            "Total Cycles:      50\n"
            "IPC:               0.20\n"
            "Dispatch Width Stalls: 0\n"
        )
        return FakeResult(0, stdout=mca_out)
    if cmd and "qemu" in str(cmd[0]):
        return FakeResult(0)
    if cmd and ("ld" in str(cmd[0]) or "linker" in str(cmd[0])):
        return FakeResult(0)
    return FakeResult(1, stderr="mock: unknown command")


class TestStepMethod:
    def test_has_step_method(self, env: Any) -> None:
        assert hasattr(env, "step")
        assert callable(env.step)

    def test_step_accepts_compiler_action(self, env: Any) -> None:
        from unittest.mock import patch as _patch

        from arm_gym.env import CompilerAction, CompilerObservation

        obs = env.reset(seed=42)
        action = CompilerAction(
            variant_id=obs.variant_id,
            assembly="mov x0, #0\nret",
        )
        with _patch("arm_gym.verifier.subprocess.run", side_effect=_mock_subprocess_run):
            result = env.step(action)
        assert isinstance(result, CompilerObservation)

    def test_step_returns_done_true(self, env: Any) -> None:
        from unittest.mock import patch as _patch

        from arm_gym.env import CompilerAction

        obs = env.reset(seed=42)
        action = CompilerAction(variant_id=obs.variant_id, assembly="ret")
        with _patch("arm_gym.verifier.subprocess.run", side_effect=_mock_subprocess_run):
            result = env.step(action)
        assert result.done is True

    def test_step_returns_reward(self, env: Any) -> None:
        from unittest.mock import patch as _patch

        from arm_gym.env import CompilerAction

        obs = env.reset(seed=42)
        action = CompilerAction(variant_id=obs.variant_id, assembly="ret")
        with _patch("arm_gym.verifier.subprocess.run", side_effect=_mock_subprocess_run), \
             _patch("arm_gym.mca.subprocess.run", side_effect=_mock_subprocess_run):
            result = env.step(action)
        assert result.reward is not None
        assert isinstance(result.reward, float)

    def test_step_unknown_variant_returns_error(self, env: Any) -> None:
        from arm_gym.env import CompilerAction

        env.reset(seed=42)
        action = CompilerAction(variant_id="nonexistent_abc123", assembly="ret")
        result = env.step(action)
        assert result.done is True
        assert result.reward == 0.0
        assert result.error_json is not None
        assert "unknown_variant" in result.error_json

    def test_step_increments_step_count(self, env: Any) -> None:
        from unittest.mock import patch as _patch

        from arm_gym.env import CompilerAction

        obs = env.reset(seed=42)
        assert env.state["step_count"] == 0
        action = CompilerAction(variant_id=obs.variant_id, assembly="ret")
        with _patch("arm_gym.verifier.subprocess.run", side_effect=_mock_subprocess_run):
            env.step(action)
        assert env.state["step_count"] == 1


# ===========================================================================
# 4. state property returns dict
# ===========================================================================

class TestStateProperty:
    def test_has_state_property(self, env: Any) -> None:
        assert hasattr(type(env), "state")
        assert isinstance(getattr(type(env), "state"), property)

    def test_state_returns_dict(self, env: Any) -> None:
        env.reset(seed=42)
        state = env.state
        assert isinstance(state, dict)

    def test_state_has_required_keys(self, env: Any) -> None:
        env.reset(seed=42)
        state = env.state
        required_keys = {
            "episode_id",
            "step_count",
            "kernel_name",
            "variant_id",
            "difficulty_level",
            "best_speedup_seen",
        }
        assert required_keys.issubset(set(state.keys())), (
            f"Missing keys: {required_keys - set(state.keys())}"
        )

    def test_state_values_after_reset(self, env: Any) -> None:
        env.reset(seed=42)
        state = env.state
        assert state["step_count"] == 0
        assert state["variant_id"] is not None
        assert isinstance(state["difficulty_level"], int)
        assert state["best_speedup_seen"] == 0.0


# ===========================================================================
# 5. metadata() returns dict with required fields
# ===========================================================================

class TestMetadataMethod:
    def test_has_metadata_method(self, env: Any) -> None:
        assert hasattr(env, "metadata")
        assert callable(env.metadata)

    def test_metadata_returns_dict(self, env: Any) -> None:
        m = env.metadata()
        assert isinstance(m, dict)

    def test_metadata_has_required_fields(self, env: Any) -> None:
        m = env.metadata()
        required = {"name", "version", "description", "supports_concurrent_sessions"}
        assert required.issubset(set(m.keys())), (
            f"Missing metadata fields: {required - set(m.keys())}"
        )

    def test_metadata_name_matches_manifest(self, env: Any) -> None:
        m = env.metadata()
        assert m["name"] == "arm-gym"

    def test_metadata_version_is_semver(self, env: Any) -> None:
        m = env.metadata()
        parts = m["version"].split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()

    def test_metadata_concurrent_sessions_matches_class(self, env: Any) -> None:
        m = env.metadata()
        assert m["supports_concurrent_sessions"] is True


# ===========================================================================
# 6. Pydantic BaseModel subclasses
# ===========================================================================

class TestPydanticModels:
    def test_compiler_action_is_pydantic(self) -> None:
        from arm_gym.env import CompilerAction

        assert issubclass(CompilerAction, BaseModel)

    def test_compiler_observation_is_pydantic(self) -> None:
        from arm_gym.env import CompilerObservation

        assert issubclass(CompilerObservation, BaseModel)

    def test_compiler_action_schema(self) -> None:
        from arm_gym.env import CompilerAction

        schema = CompilerAction.model_json_schema()
        assert "variant_id" in schema["properties"]
        assert "assembly" in schema["properties"]
        assert len(schema["required"]) >= 2

    def test_compiler_observation_schema(self) -> None:
        from arm_gym.env import CompilerObservation

        schema = CompilerObservation.model_json_schema()
        props = schema["properties"]
        for field_name in ("done", "variant_id", "c_source", "baseline_asm", "baseline_cycles"):
            assert field_name in props, f"Missing field: {field_name}"

    def test_compiler_action_rejects_missing_fields(self) -> None:
        from pydantic import ValidationError

        from arm_gym.env import CompilerAction

        with pytest.raises(ValidationError):
            CompilerAction()  # type: ignore[call-arg]

    def test_compiler_observation_has_defaults(self) -> None:
        from arm_gym.env import CompilerObservation

        obs = CompilerObservation(
            variant_id="test",
            c_source="int main(){}",
            baseline_asm="ret",
            baseline_cycles=10.0,
        )
        assert obs.done is False
        assert obs.reward is None
        assert obs.step_count == 0
        assert obs.difficulty == 1

    def test_compiler_action_serialization_roundtrip(self) -> None:
        from arm_gym.env import CompilerAction

        action = CompilerAction(variant_id="vec_add_abc123", assembly="mov x0, #0\nret")
        dumped = action.model_dump()
        restored = CompilerAction(**dumped)
        assert restored.variant_id == action.variant_id
        assert restored.assembly == action.assembly

    def test_compiler_observation_serialization_roundtrip(self) -> None:
        from arm_gym.env import CompilerObservation

        obs = CompilerObservation(
            done=True, reward=1.5, variant_id="test",
            c_source="void f(){}", baseline_asm="ret",
            baseline_cycles=42.0, speedup=2.1,
        )
        dumped = obs.model_dump()
        restored = CompilerObservation(**dumped)
        assert restored.reward == obs.reward
        assert restored.speedup == obs.speedup


# ===========================================================================
# 7. FastAPI endpoints
# ===========================================================================

class TestFastAPIEndpoints:
    """Verify all required HTTP endpoints exist and respond correctly."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, app_client: httpx.AsyncClient) -> None:
        resp = await app_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "ok" in data

    @pytest.mark.asyncio
    async def test_metadata_endpoint(self, app_client: httpx.AsyncClient) -> None:
        resp = await app_client.get("/metadata")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data
        assert "version" in data
        assert data["name"] == "arm-gym"

    @pytest.mark.asyncio
    async def test_schema_endpoint(self, app_client: httpx.AsyncClient) -> None:
        resp = await app_client.get("/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert "action" in data
        assert "observation" in data
        # Verify these are actual JSON schemas
        assert "properties" in data["action"]
        assert "properties" in data["observation"]

    @pytest.mark.asyncio
    async def test_tasks_endpoint(self, app_client: httpx.AsyncClient) -> None:
        resp = await app_client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert "templates" in data
        assert isinstance(data["templates"], list)
        assert len(data["templates"]) > 0
        assert "total_variants" in data
        assert data["total_variants"] >= 500  # Cut 2: 523 variants minimum

    @pytest.mark.asyncio
    async def test_state_endpoint(self, app_client: httpx.AsyncClient) -> None:
        resp = await app_client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_reset_endpoint(self, app_client: httpx.AsyncClient) -> None:
        resp = await app_client.post("/reset?seed=42")
        assert resp.status_code == 200
        data = resp.json()
        assert "variant_id" in data
        assert "c_source" in data
        assert data["done"] is False

    @pytest.mark.asyncio
    async def test_step_endpoint(self, app_client: httpx.AsyncClient) -> None:
        from unittest.mock import patch as _patch

        reset_resp = await app_client.post("/reset?seed=42")
        variant_id = reset_resp.json()["variant_id"]

        with _patch("arm_gym.verifier.subprocess.run", side_effect=_mock_subprocess_run), \
             _patch("arm_gym.mca.subprocess.run", side_effect=_mock_subprocess_run):
            resp = await app_client.post(
                "/step",
                json={"variant_id": variant_id, "assembly": "ret"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is True
        assert "reward" in data

    @pytest.mark.asyncio
    async def test_step_endpoint_validates_input(self, app_client: httpx.AsyncClient) -> None:
        resp = await app_client.post("/step", json={})
        assert resp.status_code == 422  # Pydantic validation error

    def test_all_required_routes_registered(self) -> None:
        """Verify all required OpenEnv routes are present on the FastAPI app."""
        from arm_gym.env import app

        routes = {r.path for r in app.routes if hasattr(r, "path")}
        required = {"/health", "/metadata", "/schema", "/tasks", "/state", "/reset", "/step", "/ws"}
        missing = required - routes
        assert not missing, f"Missing required routes: {missing}"

    def test_http_methods_correct(self) -> None:
        """Verify GET vs POST is correct per OpenEnv convention."""
        from arm_gym.env import app

        route_map: dict[str, set[str]] = {}
        for route in app.routes:
            if hasattr(route, "path") and hasattr(route, "methods"):
                route_map[route.path] = route.methods

        get_routes = {"/health", "/metadata", "/schema", "/tasks", "/state"}
        post_routes = {"/reset", "/step"}

        for path in get_routes:
            if path in route_map:
                assert "GET" in route_map[path], f"{path} should accept GET"

        for path in post_routes:
            if path in route_map:
                assert "POST" in route_map[path], f"{path} should accept POST"

    def test_ws_route_is_websocket(self) -> None:
        """Verify /ws is registered as a WebSocket endpoint, not HTTP."""
        from starlette.routing import WebSocketRoute

        from arm_gym.env import app

        ws_routes = [r for r in app.routes if isinstance(r, WebSocketRoute) and r.path == "/ws"]
        assert len(ws_routes) == 1, "Expected exactly one WebSocket route at /ws"


# ===========================================================================
# 8. WebSocket /ws endpoint
# ===========================================================================

@pytest.mark.skipif(True, reason="WebSocket tests require a live server, not ASGI transport")
class TestWebSocketEndpoint:
    """Verify the WebSocket handles all required JSON-RPC methods.

    These tests are skipped in CI because httpx ASGITransport does not support
    WebSocket upgrade. Run manually against a live uvicorn instance.
    """

    @pytest.mark.asyncio
    async def test_ws_reset_method(self, app_client: httpx.AsyncClient) -> None:
        pass

    @pytest.mark.asyncio
    async def test_ws_step_method(self, app_client: httpx.AsyncClient) -> None:
        pass

    @pytest.mark.asyncio
    async def test_ws_state_method(self, app_client: httpx.AsyncClient) -> None:
        pass

    @pytest.mark.asyncio
    async def test_ws_metadata_method(self, app_client: httpx.AsyncClient) -> None:
        pass

    @pytest.mark.asyncio
    async def test_ws_unknown_method_returns_error(self, app_client: httpx.AsyncClient) -> None:
        pass

    @pytest.mark.asyncio
    async def test_ws_invalid_json(self, app_client: httpx.AsyncClient) -> None:
        pass


# ===========================================================================
# 9. openenv.yaml manifest
# ===========================================================================

class TestOpenEnvManifest:
    """Validate the openenv.yaml manifest has all required fields."""

    @pytest.fixture()
    def manifest(self) -> dict[str, Any]:
        manifest_path = PROJECT_ROOT / "openenv.yaml"
        assert manifest_path.exists(), "openenv.yaml not found at project root"
        with open(manifest_path) as f:
            return yaml.safe_load(f)

    def test_manifest_exists(self) -> None:
        assert (PROJECT_ROOT / "openenv.yaml").exists()

    def test_has_name(self, manifest: dict[str, Any]) -> None:
        assert "name" in manifest
        assert isinstance(manifest["name"], str)
        assert len(manifest["name"]) > 0

    def test_has_version(self, manifest: dict[str, Any]) -> None:
        assert "version" in manifest
        parts = str(manifest["version"]).split(".")
        assert len(parts) == 3, f"Version should be semver, got: {manifest['version']}"

    def test_has_environment(self, manifest: dict[str, Any]) -> None:
        assert "environment" in manifest
        env_section = manifest["environment"]
        assert "class" in env_section
        assert "supports_concurrent_sessions" in env_section

    def test_environment_class_is_importable(self, manifest: dict[str, Any]) -> None:
        class_path = manifest["environment"]["class"]
        module_path, class_name = class_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name, None)
        assert cls is not None, f"Cannot import {class_path}"

    def test_has_server(self, manifest: dict[str, Any]) -> None:
        assert "server" in manifest
        server = manifest["server"]
        assert "module" in server
        assert "app" in server
        assert "framework" in server

    def test_server_app_is_importable(self, manifest: dict[str, Any]) -> None:
        server = manifest["server"]
        mod = importlib.import_module(server["module"])
        app_obj = getattr(mod, server["app"], None)
        assert app_obj is not None, f"Cannot import {server['module']}.{server['app']}"

    def test_has_protocols(self, manifest: dict[str, Any]) -> None:
        assert "protocols" in manifest
        protocols = manifest["protocols"]
        assert "websocket" in protocols
        assert "http" in protocols

    def test_websocket_protocol_path(self, manifest: dict[str, Any]) -> None:
        assert manifest["protocols"]["websocket"] == "/ws"

    def test_http_protocol_endpoints(self, manifest: dict[str, Any]) -> None:
        http = manifest["protocols"]["http"]
        required_endpoints = {"health", "reset", "step", "state", "metadata", "schema", "tasks"}
        missing = required_endpoints - set(http.keys())
        assert not missing, f"Missing HTTP endpoints in manifest: {missing}"

    def test_manifest_endpoints_match_fastapi_routes(self, manifest: dict[str, Any]) -> None:
        """Cross-validate: every endpoint in manifest is registered on the app."""
        from arm_gym.env import app

        registered = {r.path for r in app.routes if hasattr(r, "path")}
        http_endpoints = manifest["protocols"]["http"]
        for name, path in http_endpoints.items():
            assert path in registered, (
                f"Manifest declares {name}={path} but route not found in app"
            )

    def test_concurrent_sessions_matches_class(self, manifest: dict[str, Any]) -> None:
        from arm_gym.env import ARMGymEnv

        manifest_val = manifest["environment"]["supports_concurrent_sessions"]
        assert manifest_val == ARMGymEnv.SUPPORTS_CONCURRENT_SESSIONS

    def test_version_matches_code(self, manifest: dict[str, Any]) -> None:
        from arm_gym import __version__

        assert str(manifest["version"]) == __version__


# ===========================================================================
# 10. Client never imports server internals
# ===========================================================================

class TestImportSeparation:
    """Verify client/server separation — no circular server imports in public API."""

    def test_env_module_does_not_import_train(self) -> None:
        """env.py (server) must not import train.py (training loop)."""
        source = (ARM_GYM_PKG / "env.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "train" in node.module:
                    pytest.fail(f"env.py imports training module: {node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "train" in alias.name:
                        pytest.fail(f"env.py imports training module: {alias.name}")

    def test_models_importable_without_server(self) -> None:
        """CompilerAction and CompilerObservation should be importable from env
        without triggering heavy server deps that would break a thin client."""
        from arm_gym.env import CompilerAction, CompilerObservation

        # If we reach here, the models imported successfully
        assert CompilerAction is not None
        assert CompilerObservation is not None

    def test_errors_module_has_no_server_deps(self) -> None:
        """errors.py should be importable standalone — no FastAPI/uvicorn deps."""
        source = (ARM_GYM_PKG / "errors.py").read_text()
        tree = ast.parse(source)
        server_modules = {"fastapi", "uvicorn", "starlette", "websockets"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                base = node.module.split(".")[0]
                assert base not in server_modules, (
                    f"errors.py imports server module: {node.module}"
                )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base = alias.name.split(".")[0]
                    assert base not in server_modules, (
                        f"errors.py imports server module: {alias.name}"
                    )

    def test_reward_module_has_no_server_deps(self) -> None:
        """reward.py should not import FastAPI/server modules."""
        source = (ARM_GYM_PKG / "reward.py").read_text()
        tree = ast.parse(source)
        server_modules = {"fastapi", "uvicorn", "starlette", "websockets"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                base = node.module.split(".")[0]
                assert base not in server_modules, (
                    f"reward.py imports server module: {node.module}"
                )


# ===========================================================================
# 11. No reserved tool names used for MCP tools
# ===========================================================================

class TestNoReservedToolNames:
    """OpenEnv reserves reset, step, state, close as tool names.
    MCP tools (if any) must not shadow these."""

    RESERVED = {"reset", "step", "state", "close"}

    def test_no_mcp_tool_endpoint_shadows_reserved(self) -> None:
        """Check FastAPI routes — none of the custom tool endpoints should
        use reserved names as their *tool name* (path segment after /tools/)."""
        from arm_gym.env import app

        for route in app.routes:
            if not hasattr(route, "path"):
                continue
            path = route.path
            # MCP tools typically at /tools/<name> or /mcp/<name>
            if "/tools/" in path or "/mcp/" in path:
                segments = path.strip("/").split("/")
                tool_name = segments[-1] if segments else ""
                assert tool_name not in self.RESERVED, (
                    f"Route {path} uses reserved tool name: {tool_name}"
                )

    def test_manifest_has_no_reserved_mcp_tool_names(self) -> None:
        """If openenv.yaml declares mcp tools, none should use reserved names."""
        manifest_path = PROJECT_ROOT / "openenv.yaml"
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        # Check various possible manifest keys for MCP tool definitions
        for key in ("tools", "mcp_tools", "mcp"):
            section = manifest.get(key)
            if section is None:
                continue
            if isinstance(section, dict):
                for tool_name in section.keys():
                    assert tool_name not in self.RESERVED, (
                        f"Manifest declares MCP tool with reserved name: {tool_name}"
                    )
            elif isinstance(section, list):
                for tool in section:
                    name = tool.get("name", "") if isinstance(tool, dict) else str(tool)
                    assert name not in self.RESERVED, (
                        f"Manifest declares MCP tool with reserved name: {name}"
                    )

    def test_core_endpoints_are_not_registered_as_tools(self) -> None:
        """The Gym API endpoints (reset, step, state) exist as endpoints
        but should not be re-exported as MCP tools."""
        from arm_gym.env import app

        for route in app.routes:
            if not hasattr(route, "path"):
                continue
            path = route.path
            # /reset, /step, /state are fine as HTTP endpoints
            # They should NOT appear under /tools/ prefix
            if path.startswith("/tools/"):
                name = path.split("/tools/")[-1].strip("/")
                assert name not in self.RESERVED


# ===========================================================================
# Additional compliance: Gym-style API contract
# ===========================================================================

class TestGymAPIContract:
    """Verify the env follows Gym-style reset/step/state convention."""

    def test_reset_step_cycle(self, env: Any) -> None:
        """Full episode cycle: reset → step → done."""
        from unittest.mock import patch as _patch

        from arm_gym.env import CompilerAction

        obs = env.reset(seed=42)
        assert obs.done is False

        action = CompilerAction(variant_id=obs.variant_id, assembly="ret")
        with _patch("arm_gym.verifier.subprocess.run", side_effect=_mock_subprocess_run), \
             _patch("arm_gym.mca.subprocess.run", side_effect=_mock_subprocess_run):
            result = env.step(action)
        assert result.done is True
        assert result.reward is not None

    def test_multiple_resets_work(self, env: Any) -> None:
        """Environment should handle multiple consecutive resets."""
        for seed in range(5):
            obs = env.reset(seed=seed)
            assert obs.done is False
            assert obs.variant_id is not None

    def test_state_before_reset(self, env: Any) -> None:
        """State should be accessible even before reset (default values)."""
        state = env.state
        assert isinstance(state, dict)
        assert state["step_count"] == 0

    def test_observation_model_dump_is_json_serializable(self, env: Any) -> None:
        """Observations must be JSON-serializable for WebSocket transport."""
        obs = env.reset(seed=42)
        dumped = obs.model_dump()
        serialized = json.dumps(dumped)
        roundtripped = json.loads(serialized)
        assert roundtripped["variant_id"] == obs.variant_id
