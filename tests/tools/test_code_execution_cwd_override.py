"""code_execution_tool's environment builder: override lookup and cwd guard.

Two defects, and fixing either alone is wrong.

1. The lookup read only the COLLAPSED container id
   (``_resolve_container_task_id``). A CWD-only override is registered under
   the RAW session id and deliberately collapses so sessions share one
   container, so this site silently dropped it while ``terminal_tool`` and
   ``file_tools`` -- both of which go through ``resolve_task_overrides`` --
   honoured it. That helper's docstring calls itself "the single source of
   that lookup so the terminal and file layers can't drift apart"; this file
   was the drift.

2. The site had no cwd guard at all -- not even the container one the other
   two layers run. That looked harmless only because of defect 1. It was not:
   an override carrying an isolation key (``env_type`` or ``*_image``, as RL
   and benchmark harnesses register) keeps the raw id, so a host path already
   reached ``docker run -w <host path>`` on exactly the rollouts that ask for
   their own sandbox.

Fixing 1 without 2 would widen the leak: the site would start finding host
paths it used to miss. These tests pin both.
"""

import tools.code_execution_tool as cet
import tools.terminal_tool as tt


def _config(env_type="docker", cwd="/root"):
    return {
        "env_type": env_type,
        "docker_image": "pytorch/pytorch:latest", "singularity_image": "",
        "modal_image": "", "daytona_image": "",
        "cwd": cwd, "timeout": 180, "lifetime_seconds": 300,
        "container_cpu": 1, "container_memory": 5120, "container_disk": 51200,
        "container_persistent": True, "docker_volumes": [],
        "docker_run_as_host_user": False, "docker_network": True,
        "modal_mode": "auto",
        "ssh_host": "", "ssh_user": "", "ssh_port": 22, "ssh_key": "",
    }


def _build(monkeypatch, task_id, overrides, env_type="docker", config_cwd="/root"):
    """Drive _get_or_create_env and return the cwd handed to the builder."""
    captured = {}
    cfg = _config(env_type, config_cwd)

    class _DummyEnv:
        cwd = config_cwd

    def fake_create_environment(env_type, image, cwd, timeout, **kwargs):
        captured["cwd"] = cwd
        return _DummyEnv()

    monkeypatch.setattr(tt, "_get_env_config", lambda: cfg)
    monkeypatch.setattr(tt, "_create_environment", fake_create_environment)
    monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(tt, "_active_environments", {})
    monkeypatch.setattr(tt, "_last_activity", {})

    tt.register_task_env_overrides(task_id, overrides)
    try:
        cet._get_or_create_env(task_id)
    finally:
        tt.clear_task_env_overrides(task_id)
        tt.clear_session_cwd(task_id)
        tt._active_environments.pop(task_id, None)
        tt._active_environments.pop("default", None)
    return captured.get("cwd")


class TestOverrideLookupMatchesTheOtherLayers:
    def test_cwd_only_override_is_found(self, monkeypatch):
        # Collapses to "default"; the old lookup missed it entirely.
        cwd = _build(monkeypatch, "sess-cwd-only", {"cwd": "/workspace/task42"})
        assert cwd == "/workspace/task42"

    def test_the_other_layers_agree(self, monkeypatch):
        # Pin the equivalence rather than the mechanism: whatever
        # resolve_task_overrides returns is what this site must see.
        tt.register_task_env_overrides("sess-agree", {"cwd": "/workspace/x"})
        try:
            assert tt.resolve_task_overrides("sess-agree").get("cwd") == "/workspace/x"
            assert _build(monkeypatch, "sess-agree2", {"cwd": "/workspace/x"}) == "/workspace/x"
        finally:
            tt.clear_task_env_overrides("sess-agree")


class TestHostCwdDoesNotReachTheContainerBuilder:
    def test_cwd_only_host_override_is_sanitized(self, monkeypatch):
        # Newly reachable because of the lookup fix -- and therefore newly
        # dangerous without the guard.
        cwd = _build(monkeypatch, "sess-host", {"cwd": r"C:\Users\someuser"})
        assert cwd == "/root"

    def test_isolation_keyed_host_override_is_sanitized(self, monkeypatch):
        # Reachable even before the lookup fix: an isolation key keeps the raw
        # id. This is the half that was already leaking.
        cwd = _build(monkeypatch, "sess-iso",
                     {"cwd": "/home/someuser/project", "docker_image": "alpine"})
        assert cwd == "/root"

    def test_drive_path_host_override_is_sanitized(self, monkeypatch):
        cwd = _build(monkeypatch, "sess-drive", {"cwd": r"D:\hermes\kanban"})
        assert cwd == "/root"

    def test_valid_container_override_passes_through(self, monkeypatch):
        cwd = _build(monkeypatch, "sess-ok", {"cwd": "/workspace/task42"})
        assert cwd == "/workspace/task42"

    def test_non_container_backend_is_untouched(self, monkeypatch):
        # The guard is container-scoped; local must not be narrowed by it.
        cwd = _build(monkeypatch, "sess-local", {"cwd": "/home/someuser"},
                     env_type="local", config_cwd="/home/hermes")
        assert cwd == "/home/someuser"
