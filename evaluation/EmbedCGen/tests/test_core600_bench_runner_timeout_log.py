import subprocess
from pathlib import Path

from scripts import core600_bench_runner as runner


def test_simulate_elf_preserves_renode_output_on_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    renode_bin = tmp_path / "renode"
    renode_platform = tmp_path / "platform.repl"
    elf_file = tmp_path / "case.elf"
    renode_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    renode_platform.write_text("platform", encoding="utf-8")
    elf_file.write_text("elf", encoding="utf-8")
    renode_log_dir = tmp_path / "renode_logs"

    monkeypatch.setattr(runner, "RENODE_BIN", renode_bin)
    monkeypatch.setattr(runner, "RENODE_PLATFORM", renode_platform)
    monkeypatch.setattr(runner, "SIM_TIMEOUT_FLOOR", 11)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=kwargs["timeout"],
            output=b"partial renode stdout\nUSART: EDGE_CANDIDATE\n",
            stderr=b"partial renode stderr\n",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    ok, _elapsed, status, category, reason, verdict = runner.simulate_elf(
        elf_file,
        19,
        renode_log_dir,
        sim_seconds=1,
        case=None,
        semantic_oracle=False,
        renode_timeout_floor=11,
    )

    assert ok is False
    assert status == "sim_timeout"
    assert category == "env"
    assert reason == "sim_timeout"
    assert verdict is None
    assert (renode_log_dir / "core600_19.resc").exists()

    log_text = (renode_log_dir / "core600_19.log").read_text(encoding="utf-8")
    assert "---RENODE TIMEOUT---" in log_text
    assert "timeout_seconds=11" in log_text
    assert "partial renode stdout" in log_text
    assert "EDGE_CANDIDATE" in log_text
    assert "partial renode stderr" in log_text


def test_core600_platform_copy_disables_svd_by_default(tmp_path: Path, monkeypatch) -> None:
    renode_platform = tmp_path / "platform.repl"
    local_svd = tmp_path / "STM32F40x.svd"
    renode_platform.write_text(
        "sysbus:\n"
        "    init:\n"
        "        ApplySVD @https://dl.antmicro.com/projects/renode/svd/STM32F40x.svd.gz\n",
        encoding="utf-8",
    )
    local_svd.write_text("<device/>", encoding="utf-8")
    monkeypatch.setattr(runner, "RENODE_PLATFORM", renode_platform)
    monkeypatch.setattr(runner, "LOCAL_STM32F40X_SVD", local_svd)
    monkeypatch.delenv("CORE600_RENODE_APPLY_SVD", raising=False)

    platform_copy = runner._core600_renode_platform(tmp_path / "renode_logs")

    text = platform_copy.read_text(encoding="utf-8")
    assert "https://dl.antmicro.com/projects/renode/svd/STM32F40x.svd.gz" not in text
    assert f"ApplySVD @{local_svd}" not in text
    assert "disables ApplySVD" in text


def test_simulate_elf_can_enable_peripheral_access_trace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    renode_bin = tmp_path / "renode"
    renode_platform = tmp_path / "platform.repl"
    elf_file = tmp_path / "case.elf"
    renode_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    renode_platform.write_text("platform", encoding="utf-8")
    elf_file.write_text("elf", encoding="utf-8")
    renode_log_dir = tmp_path / "renode_logs"
    monkeypatch.setattr(runner, "RENODE_BIN", renode_bin)
    monkeypatch.setattr(runner, "RENODE_PLATFORM", renode_platform)

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    runner.simulate_elf(
        elf_file,
        31,
        renode_log_dir,
        sim_seconds=1,
        case=None,
        semantic_oracle=False,
        trace_peripheral_access=True,
    )

    resc = (renode_log_dir / "core600_31.resc").read_text(encoding="utf-8")
    assert "sysbus LogAllPeripheralsAccess true" in resc


def test_core600_platform_copy_can_use_repo_local_svd(tmp_path: Path, monkeypatch) -> None:
    renode_platform = tmp_path / "platform.repl"
    local_svd = tmp_path / "STM32F40x.svd"
    renode_platform.write_text(
        "sysbus:\n"
        "    init:\n"
        "        ApplySVD @https://dl.antmicro.com/projects/renode/svd/STM32F40x.svd.gz\n",
        encoding="utf-8",
    )
    local_svd.write_text("<device/>", encoding="utf-8")
    monkeypatch.setattr(runner, "RENODE_PLATFORM", renode_platform)
    monkeypatch.setattr(runner, "LOCAL_STM32F40X_SVD", local_svd)
    monkeypatch.setenv("CORE600_RENODE_APPLY_SVD", "1")

    platform_copy = runner._core600_renode_platform(tmp_path / "renode_logs")

    text = platform_copy.read_text(encoding="utf-8")
    assert "https://dl.antmicro.com/projects/renode/svd/STM32F40x.svd.gz" not in text
    assert f"ApplySVD @{local_svd}" in text


def test_simulate_elf_scores_complete_semantic_evidence_on_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    renode_bin = tmp_path / "renode"
    renode_platform = tmp_path / "platform.repl"
    elf_file = tmp_path / "case.elf"
    renode_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    renode_platform.write_text("platform", encoding="utf-8")
    elf_file.write_text("elf", encoding="utf-8")
    renode_log_dir = tmp_path / "renode_logs"
    monkeypatch.setattr(runner, "RENODE_BIN", renode_bin)
    monkeypatch.setattr(runner, "RENODE_PLATFORM", renode_platform)
    monkeypatch.setattr(runner, "LOCAL_STM32F40X_SVD", tmp_path / "missing.svd")

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=kwargs["timeout"],
            output="\n".join(
                [
                    "[INFO] usart2: DMA_BUSY",
                    "[INFO] usart2: HT_READY",
                    "[INFO] usart2: FRAME_OK",
                    "[WARNING] dma2: Unhandled write to offset 0x28. Tags: HTIE (0x1).",
                    'echo "CORE600_READBACK_AFTER_001_DMA1_STREAM5_CR"',
                    "sysbus: ReadDoubleWord returned 0x00000001",
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    case = {
        "oracle_contract_mode": "semantic_strict",
        "required_ordered_logs": ["DMA_BUSY", "HT_READY", "FRAME_OK"],
        "required_control_pattern": ["half_transfer_interrupt"],
        "failure_oracle": [],
        "state_readback_expectations": [
            {
                "target": "DMA1 Stream5 CR",
                "monitor_command": "sysbus ReadDoubleWord 0x40026088",
                "expected_mask": "0x1",
                "expected_relation": "bit_set",
            }
        ],
    }

    ok, _elapsed, status, category, reason, verdict = runner.simulate_elf(
        elf_file,
        23,
        renode_log_dir,
        sim_seconds=1,
        case=case,
        semantic_oracle=True,
        renode_timeout_floor=11,
    )

    assert ok is True
    assert status == "ok"
    assert category == ""
    assert reason == ""
    assert verdict["likely_correct"] is True
