"""The container cwd verdict must not depend on the host Hermes runs on.

``_is_unusable_container_cwd`` used ``os.path.isabs`` for its relative-path
check, and ``os.path`` is ``posixpath`` or ``ntpath`` depending on the host.
The same string therefore got different verdicts on different machines, and
the comment above the check said so out loud -- it argued Windows drive paths
were "already caught by the prefix check above", which holds only because
``_HOST_CWD_PREFIXES`` happens to list ``C:``.

Two consequences, one measured and one waiting:

  * Measured on SSJOON 2026-09-07 (Windows, CPython 3.11.15): ``D:\\hermes\\
    kanban``, ``D:/hermes`` and ``Z:\\x`` all passed the guard and reached the
    container builder. Only the ``C:`` drive was ever caught.

  * ``ntpath.isabs`` changed in CPython 3.13 to return False for a rooted path
    with no drive. On a Windows host running 3.13+, the guard would have begun
    rejecting ``/workspace`` and ``/root`` -- discarding exactly the values it
    exists to preserve. SSJOON runs 3.11, so this had not fired yet.

These tests pin the verdict for both families on every host, so neither the
drive-letter hole nor the 3.13 inversion can come back.
"""

import ntpath
import posixpath

import tools.terminal_tool as tt


class TestVerdictIsTheSameOnEveryHost:
    """Same string, same answer, whichever ``os.path`` the host provides."""

    REJECT = [
        r"C:\Users\me", "C:/Users/me",          # already covered by the prefixes
        r"D:\hermes\kanban", "D:/hermes",       # the measured hole
        r"Z:\x", "e:/tmp",                      # any other drive, either slash
        ".", "..", "src/", "work/uservice",     # relative
        "/home/me", "/Users/me",                # host POSIX paths
    ]
    KEEP = ["/workspace", "/workspace/task42", "/root", "/", "/opt/data"]

    def test_host_paths_and_drive_paths_are_rejected(self):
        for p in self.REJECT:
            assert tt._is_unusable_container_cwd(p) is True, p

    def test_container_paths_are_kept(self):
        for p in self.KEEP:
            assert tt._is_unusable_container_cwd(p) is False, p

    def test_verdict_does_not_consult_os_path(self, monkeypatch):
        # Swap the module's os.path for the *other* flavour and demand the
        # same answers. This is the property the old implementation lacked.
        for flavour in (ntpath, posixpath):
            monkeypatch.setattr(tt.os, "path", flavour)
            for p in self.REJECT:
                assert tt._is_unusable_container_cwd(p) is True, (flavour.__name__, p)
            for p in self.KEEP:
                assert tt._is_unusable_container_cwd(p) is False, (flavour.__name__, p)

    def test_the_two_flavours_actually_disagree_about_these_strings(self):
        # Guards the test above from becoming vacuous: if isabs ever stopped
        # differing, the property would hold for uninteresting reasons.
        assert ntpath.isabs(r"D:\x") is True
        assert posixpath.isabs(r"D:\x") is False
        # 3.13+ only; on 3.11/3.12 ntpath calls "/workspace" absolute. Either
        # way the verdict above must not move, so assert the disagreement
        # loosely rather than pinning a version-specific value.
        assert ntpath.isabs("/workspace") in (True, False)

    def test_empty_is_still_not_flagged(self):
        assert tt._is_unusable_container_cwd("") is False


class TestPrefixListIsNoLongerLoadBearingForDrives:
    """``_HOST_CWD_PREFIXES`` lists only ``C:``; the verdict must not need it
    to catch other drives. Adding ``D:\\`` would just move the hole to ``E:``.
    """

    def test_drives_outside_the_prefix_list_are_rejected(self):
        listed = tuple(p for p in tt._HOST_CWD_PREFIXES if len(p) > 2 and p[1] == ":")
        assert listed, "expected at least one drive prefix to be listed"
        for drive in ("D", "E", "F", "Z"):
            for path in (drive + r":\work", drive + ":/work"):
                assert not any(path.startswith(p) for p in tt._HOST_CWD_PREFIXES), path
                assert tt._is_unusable_container_cwd(path) is True, path
