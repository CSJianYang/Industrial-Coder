#!/usr/bin/env python3
"""Core600 Large-Scale Multi-Model Benchmark Runner

Runs 600 high-difficulty embedded C code generation test cases against
multiple models via SiliconFlow API, with ARM GCC compilation and
optional Renode simulation.

Usage:
    # Single model, full 600
    python scripts/core600_bench_runner.py \\
        --model "Pro/zai-org/GLM-5.1" \\
        --cases data/benchmarks/core600_eval_set.json \\
        --output output_benchmark/core600/GLM-5.1

    # Multiple models in parallel (via AGENT TEAM)
    python scripts/core600_bench_runner.py --model ... --start 0 --end 300 &
    python scripts/core600_bench_runner.py --model ... --start 300 --end 600 &

Supports checkpoint/resume: if --output/results.jsonl exists, already-tested
cases are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, List, Optional, Tuple

import requests

from cgen.hardware_contract import (
    BENCHMARK_HARDWARE_CONTEXT_MEMBER_GUIDANCE,
    DMA_ARCHITECTURE_GUIDANCE,
)
from cgen.core600_hardware_context import (
    BASE_CONTEXT,
    OPTIONAL_SLICES,
    build_hw_context as build_core600_hardware_context,
)

# ── Project paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = PROJECT_ROOT / "templates"
TEMPLATE_LINKER = TEMPLATE_DIR / "STM32F407xG.ld"
TEMPLATE_STARTUP = TEMPLATE_DIR / "startup_stm32f407xx.s"
RENODE_BIN = PROJECT_ROOT / "tools/renode/renode"
RENODE_PLATFORM = PROJECT_ROOT / "tools/renode/platforms/cpus/stm32f4.repl"
LOCAL_STM32F40X_SVD = PROJECT_ROOT / "data/svd/STMicro/STM32F40x.svd"
GCC_BIN = PROJECT_ROOT / "tools/arm-gnu-toolchain-14.3.rel1-x86_64-arm-none-eabi/bin/arm-none-eabi-gcc"

# ── LLM API ────────────────────────────────────────────────────
SF_API_URL = os.getenv(
    "SF_API_URL", "https://api.siliconflow.cn/v1/chat/completions"
)
SF_API_KEY = os.getenv("SF_API_KEY", "")
DEFAULT_CONCURRENCY = 1
CORE600_CHECKPOINT_PROTOTYPE = "void CORE600_CHECKPOINT(uint32_t checkpoint_id)"

# ── Evaluation constants ──────────────────────────────────────────────
CHIP = "STM32F407VGT6"
MAX_TOKENS = int(os.getenv("CORE600_MAX_TOKENS", "16384"))
LLM_TIMEOUT = int(os.getenv("CORE600_LLM_TIMEOUT", "600"))
LLM_MAX_RETRIES = int(os.getenv("CORE600_LLM_MAX_RETRIES", "3"))
HTTP_FAIL_FAST_STATUS_CODES = {400, 401, 403, 404}
COMPILE_TIMEOUT = 30
SIM_TIMEOUT_FLOOR = 60
SIM_SECONDS = float(os.getenv("CORE600_SIM_SECONDS", "1"))
RENODE_TIMEOUT_FLOOR = int(os.getenv("CORE600_RENODE_TIMEOUT_FLOOR", str(SIM_TIMEOUT_FLOOR)))
RENODE_TRACE_PERIPHERAL_ACCESS = os.getenv("CORE600_RENODE_TRACE_PERIPHERAL_ACCESS", "").lower() in {
    "1",
    "true",
    "yes",
}
FAMILY_ORDER = ["HF-A1", "HF-B1", "HF-C1", "HF-C2", "HF-D1", "HF-E1", "HF-E2", "HF-E3", "HF-E4"]
RUN_CONTEXT_FILENAME = "run_context.json"
NEXT200_POLICY_SUPPORT_FLAG = "next200_policy_candidate_support_enabled"
NEXT200_POLICY_SURFACE_STABLE_SHA_KEY = "next200_policy_candidate_surface_stable_sha256"
NEXT200_POLICY_CANDIDATE_SURFACE_STABLE_SHA256 = "cb20ee64d1f58ea61785b00ff9c8b3a407a749db26607ccc0883352478d98b88"
NEXT200_VALIDATED_SUPPORT_FLAG = "next200_validated_candidate_support_enabled"
NEXT200_VALIDATED_RUNNER_SUPPORT_STABLE_SHA_KEY = "next200_validated_runner_support_stable_sha256"
NEXT200_VALIDATED_NEGATIVE_FIXTURE_STABLE_SHA_KEY = "next200_validated_negative_fixture_stable_sha256"
NEXT200_VALIDATED_RUNNER_SUPPORT_STABLE_SHA256 = "2050751baa6b92c6b80c159b58ab677a9151c7972c771d1a750f3b0f0a7a18ab"
NEXT200_VALIDATED_NEGATIVE_FIXTURE_STABLE_SHA256 = "bc79be94387966d38d4d832bab42c461e458082ff53c8fbdd3315541d867cc1c"
NEXT200_KNOWN_POLICY_CANDIDATE_IDS = {
    "next200_expected_idle_loop_timeout_policy_v1",
    "next200_exti_pr_observability_policy_v1",
    "next200_scb_icsr_interrupt_state_policy_v1",
    "next200_spi2_sr_stable_ready_policy_v1",
    "next200_status_polling_event_reachability_policy_v1",
    "next200_tim2_sr_blocked_timer_policy_v1",
    "next200_tim5_sr_event_policy_v1",
    "next200_timeout_postmortem_semantic_oracle_policy_v1",
}
NEXT200_VALIDATED_KNOWN_CANDIDATE_IDS = {
    "validated_candidate_loader",
    "timer_input_capture_diagnostic_demoter",
    "timeout_liveness_diagnostic_reporter",
    "validated_candidate_release_audit",
}
INFRA_INVALID_REASONS = {
    "compiler_missing",
    "renode_not_found",
    "renode_platform_missing",
    "renode_nonzero_exit",
    "sim_exception",
}
OBSERVABLE_PERIPHERAL_PATTERNS = [
    r"Write(?:DoubleWord|Word|UInt32|UInt16|Byte).*\b0x4[0-9a-f]{7}\b",
    r"\b(?:rcc|gpio[a-i]|gpioPort[A-I]|usart\d+|uart\d+|timer\d+|tim\d+|spi\d+|i2c\d+|dma\d+|adc\d+|crc|wwdg|syscfg|exti|nvic)\b:.*\bWrite",
    r"\b(?:rcc|gpio[a-i]|gpioPort[A-I]|usart\d+|uart\d+|timer\d+|tim\d+|spi\d+|i2c\d+|dma\d+|adc\d+|crc|wwdg|syscfg|exti|nvic)\b:.*\bnot supported\b",
    r"\bsysbus\b:.*Write(?:DoubleWord|Word|UInt32|UInt16|Byte)\b.*\b0x4[0-9a-f]{7}\b",
]
FAMILY_OBSERVABLE_HINTS = {
    "HF-A1": {"dma", "usart", "uart"},
    "HF-B1": {"dma", "usart", "uart"},
    "HF-C1": {"wwdg", "iwdg", "exti", "syscfg", "timer", "tim"},
    "HF-D1": {"exti", "syscfg", "timer", "tim", "usart", "uart"},
    "HF-E3": {"usart", "uart", "dma", "crc", "syscfg"},
    "HF-E4": {"timer", "tim", "gpio"},
}
TOKEN_OBSERVABLE_HINTS = {
    "dma": {"dma"},
    "usart": {"usart", "uart"},
    "uart": {"usart", "uart"},
    "serial": {"usart", "uart"},
    "watchdog": {"wwdg", "iwdg"},
    "wwdg": {"wwdg"},
    "iwdg": {"iwdg"},
    "crc": {"crc"},
    "timer": {"timer", "tim"},
    "tim": {"timer", "tim"},
    "capture": {"timer", "tim"},
    "overflow": {"timer", "tim"},
    "exti": {"exti", "syscfg"},
    "edge": {"exti", "syscfg"},
    "spi": {"spi"},
    "i2c": {"i2c"},
    "adc": {"adc"},
    "interrupt": {"nvic"},
    "irq": {"nvic"},
    "nvic": {"nvic"},
}
ORDERED_LOG_ALIASES = {
    "HT_READY": ("HT_READY", "DMA_HT"),
    "TC_READY": ("TC_READY", "DMA_TC"),
    "REENTRY_GUARDED": ("REENTRY_GUARDED", "DMA_REENTRY_GUARDED"),
}
SETUP_ONLY_NAMED_PATTERNS = [
    r"\brcc\b:.*\bWrite",
    r"\bgpio[a-i]\b:.*\bWrite",
    r"\bgpioPort[A-I]\b:.*\bWrite",
]
GENERIC_NON_SETUP_PERIPHERAL_PATTERNS = [
    r"\b(?:usart\d+|uart\d+|timer\d+|tim\d+|spi\d+|i2c\d+|dma\d+|adc\d+|crc|wwdg|iwdg|syscfg|exti|nvic)\b:.*\bWrite",
    r"\b(?:usart\d+|uart\d+|timer\d+|tim\d+|spi\d+|i2c\d+|dma\d+|adc\d+|crc|wwdg|iwdg|syscfg|exti|nvic)\b:.*\bnot supported\b",
]
SETUP_ONLY_ADDRESS_RANGES = [
    (0x40020000, 0x40022400),  # GPIOA-I
    (0x40023800, 0x40023C00),  # RCC
]

# ── Hardware context definitions ───────────────────────────────────────
# Hardware context fragments are owned by cgen.core600_hardware_context.


def _build_hw_context(requirement: str, usart_force: bool = False) -> str:
    """Build pure C hardware context (structs + #defines) for a requirement."""
    return build_core600_hardware_context(requirement, usart_force=usart_force)


def _build_system_prompt(
    requirement: str,
    required_ordered_logs: Optional[list[str]] = None,
    marker_state_bindings: Optional[list[dict[str, Any]]] = None,
    proof_contract_mode: Optional[str] = None,
) -> str:
    """Build requirement-aware system prompt with dependency-aware context injection."""
    hw_context = _build_hw_context(requirement, usart_force=bool(required_ordered_logs))

    # Build USART trace instruction if ordered logs are provided
    usart_trace_block = ""
    if required_ordered_logs:
        required_tokens = [str(token) for token in required_ordered_logs if str(token)]
        token_list = ", ".join(f'"{t}"' for t in required_tokens)
        marker_binding_by_token: dict[str, dict[str, Any]] = {}
        if isinstance(marker_state_bindings, list):
            for binding in marker_state_bindings:
                if not isinstance(binding, dict):
                    continue
                token = str(binding.get("marker") or binding.get("token") or "").strip()
                if token:
                    marker_binding_by_token[token] = binding
        if proof_contract_mode == "phase_checkpoint_v1":
            checkpoint_ids: list[int] = []
            checkpoint_names: dict[int, str] = {}
            for binding in marker_binding_by_token.values():
                try:
                    checkpoint_id = int(binding.get("checkpoint_id"))
                except (TypeError, ValueError):
                    continue
                if checkpoint_id not in checkpoint_ids:
                    checkpoint_ids.append(checkpoint_id)
                phase_name = str(binding.get("checkpoint_phase_name") or "").strip()
                if phase_name:
                    checkpoint_names[checkpoint_id] = phase_name
            checkpoint_ids.sort()
            checkpoint_calls = ", ".join(f"CORE600_CHECKPOINT({checkpoint_id})" for checkpoint_id in checkpoint_ids)
            checkpoint_lines = []
            for checkpoint_id in checkpoint_ids:
                phase_name = checkpoint_names.get(checkpoint_id, f"phase {checkpoint_id}")
                checkpoint_lines.append(f"   - after {phase_name}: call CORE600_CHECKPOINT({checkpoint_id})")
            usart_trace_block = (
                "13. You MUST use runner-provided phase checkpoints for verification.\n"
                "   Use USART2 unless the requirement explicitly needs another USART.\n"
                "   Init USART2 before the first CORE600_CHECKPOINT call: baud 115200, 8N1, TE=1, UE=1.\n"
                "   Do not declare or define trace(), trace_state(), trace_state_expect(), CORE600_CHECKPOINT(), or CORE600_CHECK_PASS; the runner already provides them.\n"
                f"   Exact provided checkpoint prototype: {CORE600_CHECKPOINT_PROTOTYPE};\n"
                "   CORE600_CHECKPOINT(n) reads the actual MMIO states bound to that phase and automatically emits CORE600_MARKER_STATE plus CORE600_CHECK_PASS only when each expected relation holds.\n"
                f"   Required phase checkpoint calls in order: {checkpoint_calls}\n"
                + "\n".join(checkpoint_lines)
                + "\n"
                "   Place each checkpoint only after completing that phase. Establish every checkpoint-required state before calling CORE600_CHECKPOINT(stage); a checkpoint called early will read the wrong MMIO state and fail.\n"
                "   Preserve checkpoint-observed state while CORE600_CHECKPOINT emits evidence, and keep final-hold/no-overwrite state valid through later checkpoints and final readback.\n"
                "   Do not frontload later-phase writes before an earlier checkpoint.\n"
                "   Preserve checkpoint transport resources after initialization: do not disable or reconfigure USART2 APB1 clocking, USART2 TE/UE/baud setup, or GPIOA PA2/PA3 USART2 alternate-function routing used for checkpoint output.\n"
                "   Do not replace checkpoints with manual token printing.\n"
                "   BEGIN/DONE wrapper text is optional diagnostic output and is not required for stable scoring.\n"
            )
            return (
                "You are an embedded C bare-metal programmer for STM32F407 (Cortex-M4).\n"
                f"TASK: {requirement}\n\n"
                "HARDWARE CONTEXT (These structs, macros and IRQn_Type are ALREADY DEFINED in the compilation environment):\n"
                f"{hw_context}\n"
                "MANDATORY RULES:\n"
                "1. DO NOT re-define ANY struct, macro or enum shown above. They are pre-included.\n"
                "2. Your code will be compiled AFTER the above definitions. Start with your own types/functions only.\n"
                "3. If a peripheral base address is missing, define ONLY that #define, not the struct.\n"
                "4. Implement 'int main(void)' with while(1) loop.\n"
                "5. NO HAL, StdPeriph or CMSIS macros. Use DIRECT bit-shift operations ONLY.\n"
                "   WRONG: RCC->AHB1ENR |= RCC_AHB1ENR_GPIOBEN;\n"
                "   RIGHT: RCC->AHB1ENR |= (1 << 1);  // GPIOBEN bit\n"
                "   WRONG: DMA1_Stream1->CR = DMA_SxCR_CHSEL_0 | DMA_SxCR_MINC;\n"
                "   RIGHT: DMA1_Stream1->CR = (4 << 25) | (1 << 10);  // channel 4, MINC\n"
                f"{DMA_ARCHITECTURE_GUIDANCE}\n"
                "6. Code only, no markdown/explanations; keep <= 350 lines.\n"
                "7. Do not redefine RCC/GPIO/NVIC/SCB/USART/TIM/EXTI/ADC/DMA/CAN/DAC/SPI/I2C/WWDG/IWDG/CRC/FLASH structs; they are already defined above.\n"
                "8. Enable NVIC and provide ISR stubs if interrupts are needed. Use IRQn_Type enum values.\n"
                f"9. {BENCHMARK_HARDWARE_CONTEXT_MEMBER_GUIDANCE}.\n"
                "   Preferred forms: GPIOx->AFR[0] / GPIOx->AFR[1], NVIC->ISER[index], NVIC->IPR[index].\n"
                "   Compatibility aliases such as GPIOx->AFRL/GPIOx->AFRH and NVIC->ISER0..ISER7 also exist, but do not invent new names.\n"
                "10. If you need size_t/offsetof, they are available through <stddef.h>; do not redeclare them.\n"
                "11. WRITE COMPLETE CODE. No '// ...' or '// TODO' placeholders.\n"
                "12. STOP after the closing '}' of main.\n"
                "CRITICAL for SIMULATION:\n"
                "- Do NOT busy-wait for hardware flags (e.g., EOC/ADON) in init. Set bit and proceed.\n"
                "- In main while(1) loop, add delay (~10000 cycles) or flag check to avoid livelock.\n"
                f"{usart_trace_block}"
            )
        trace_calls: list[str] = []
        marker_binding_lines: list[str] = []
        for token in required_tokens:
            binding = marker_binding_by_token.get(token)
            if binding:
                address = str(binding.get("address") or binding.get("monitor_address") or "").strip()
                mask = str(binding.get("expected_mask") or binding.get("mask") or "").strip()
                target = str(binding.get("target") or "").strip()
                relation = str(binding.get("expected_relation") or "bit_set").strip()
                expected_nonzero = "0u" if relation in {"bit_clear", "zero"} else "1u"
                helper_call = str(binding.get("helper_call") or "").strip()
                if helper_call == "trace_state_expect" or relation in {"bit_clear", "zero"}:
                    trace_calls.append(f'trace_state_expect("{token}", {address}, {mask}, {expected_nonzero});')
                else:
                    trace_calls.append(f'trace_state("{token}", {address}, {mask});')
                marker_binding_lines.append(
                    f'   - {token}: expected relation {relation}; call the matching helper immediately after satisfying {target}.'
                )
            else:
                trace_calls.append(f'trace("{token}");')
        trace_call_checklist = " ".join(trace_calls)
        marker_binding_block = ""
        if marker_binding_lines:
            marker_binding_block = (
                "   Marker-state binding is active for selected markers. The compilation environment provides "
                "trace_state(const char *token, uintptr_t address, uint32_t mask); do not redefine it.\n"
                "   trace_state reads the actual MMIO register at address and prints the token plus the masked readback evidence.\n"
                + "\n".join(marker_binding_lines)
                + "\n"
            )
        if proof_contract_mode == "helper_check_pass_v2":
            usart_trace_block = (
                "13. You MUST use runner-provided firmware self-readback checks for verification.\n"
                "   Use USART2 unless the requirement explicitly needs another USART.\n"
                "   Init USART2 before the first trace_state call: baud 115200, 8N1, TE=1, UE=1.\n"
                "   Do not declare or define trace(), trace_state(), or trace_state_expect(); the runner already provides their prototypes and implementations.\n"
                "   trace_state(token,address,mask) proves masked nonzero state. trace_state_expect(token,address,mask,0u) proves masked zero/unset state, and trace_state_expect(...,1u) proves masked nonzero state.\n"
                "   The helpers read actual MMIO and automatically emit CORE600_MARKER_STATE plus CORE600_CHECK_PASS only when the expected relation holds.\n"
                f"   Required helper checks in order: {token_list}\n"
                f"   Helper call checklist: {trace_call_checklist}\n"
                f"{marker_binding_block}"
                "   Call each helper at the exact requested phase: before a later-phase write for unset preconditions, and immediately after writes for phase completion/hold checks.\n"
                "   Do not replace helper calls with manual token printing.\n"
                "   BEGIN/DONE wrapper text is optional diagnostic output and is not required for stable scoring.\n"
            )
        else:
            usart_trace_block = (
                "13. You MUST print state-transition markers via USART for verification.\n"
                "   Use USART2 unless the requirement explicitly needs another USART.\n"
                "   Init the USART with: baud 115200, 8N1, TE=1, UE=1.\n"
                "   Use a concrete helper, not a placeholder: void trace(const char *s) { while(!(USART2->SR & (1<<7))); for(const char*p=s;*p;p++){while(!(USART2->SR&(1<<7)));USART2->DR=*p;} while(!(USART2->SR&(1<<7)));USART2->DR='\\r';while(!(USART2->SR&(1<<7)));USART2->DR='\\n';}\n"
                f"   Required markers (print in this sequence when the corresponding state is reached): {token_list}\n"
                "   Verification fails unless every exact marker below appears in USART output in order.\n"
                f"   Trace call checklist: {trace_call_checklist}\n"
                f"{marker_binding_block}"
                "   Place each trace() call immediately after the state transition occurs.\n"
                "   Every required marker above must appear as an exact trace(\"TOKEN\") string literal in the source.\n"
                "   For marker-state-bound markers, the exact token must appear in trace_state(\"TOKEN\", address, mask).\n"
                "   Do not rename, abbreviate, skip, merge, or print markers only in comments.\n"
                "   Do not invent enum/state identifiers from marker names unless you also declare them exactly.\n"
                "   Before finalizing the code, check that no marker from the required list is missing from trace() calls.\n"
            )

    return (
        "You are an embedded C bare-metal programmer for STM32F407 (Cortex-M4).\n"
        f"TASK: {requirement}\n\n"
        "HARDWARE CONTEXT (These structs, macros and IRQn_Type are ALREADY DEFINED in the compilation environment):\n"
        f"{hw_context}\n"
        "MANDATORY RULES:\n"
        "1. DO NOT re-define ANY struct, macro or enum shown above. They are pre-included.\n"
        "2. Your code will be compiled AFTER the above definitions. Start with your own types/functions only.\n"
        "3. If a peripheral base address is missing, define ONLY that #define, not the struct.\n"
        "4. Implement 'int main(void)' with while(1) loop.\n"
        "5. NO HAL, StdPeriph or CMSIS macros. Use DIRECT bit-shift operations ONLY.\n"
        "   WRONG: RCC->AHB1ENR |= RCC_AHB1ENR_GPIOBEN;\n"
        "   RIGHT: RCC->AHB1ENR |= (1 << 1);  // GPIOBEN bit\n"
        "   WRONG: DMA1_Stream1->CR = DMA_SxCR_CHSEL_0 | DMA_SxCR_MINC;\n"
        "   RIGHT: DMA1_Stream1->CR = (4 << 25) | (1 << 10);  // channel 4, MINC\n"
        f"{DMA_ARCHITECTURE_GUIDANCE}\n"
        "6. Code only, no markdown/explanations; keep <= 350 lines.\n"
        "7. Do not redefine RCC/GPIO/NVIC/SCB/USART/TIM/EXTI/ADC/DMA/CAN/DAC/SPI/I2C/WWDG/IWDG/CRC/FLASH structs; they are already defined above.\n"
        "8. Enable NVIC and provide ISR stubs if interrupts are needed. Use IRQn_Type enum values.\n"
        f"9. {BENCHMARK_HARDWARE_CONTEXT_MEMBER_GUIDANCE}.\n"
        "   Preferred forms: GPIOx->AFR[0] / GPIOx->AFR[1], NVIC->ISER[index], NVIC->IPR[index].\n"
        "   Compatibility aliases such as GPIOx->AFRL/GPIOx->AFRH and NVIC->ISER0..ISER7 also exist, but do not invent new names.\n"
        "10. If you need size_t/offsetof, they are available through <stddef.h>; do not redeclare them.\n"
        "11. WRITE COMPLETE CODE. No '// ...' or '// TODO' placeholders.\n"
        "12. STOP after the closing '}' of main.\n"
        "CRITICAL for SIMULATION:\n"
        "- Do NOT busy-wait for hardware flags (e.g., EOC/ADON) in init. Set bit and proceed.\n"
        "- In main while(1) loop, add delay (~10000 cycles) or flag check to avoid livelock.\n"
        f"{usart_trace_block}"
    )


def _requirement_with_generation_prompt_patch(case: dict[str, Any]) -> str:
    requirement = str(case.get("requirement") or "")
    patch = case.get("generation_prompt_patch")
    if isinstance(patch, str):
        clean_patch = patch.strip()
        if not clean_patch:
            return requirement
        return requirement + "\n\nRESIDUAL PROBE PROMPT PATCH:\n- " + clean_patch
    if not isinstance(patch, dict):
        return requirement
    lines = patch.get("lines")
    if not isinstance(lines, list):
        return requirement
    clean_lines = [str(line).strip() for line in lines if str(line).strip()]
    if not clean_lines:
        return requirement
    return requirement + "\n\nRESIDUAL PROBE PROMPT PATCH:\n" + "\n".join(f"- {line}" for line in clean_lines)


def _checkpoint_helper_source(
    marker_state_bindings: Optional[list[dict[str, Any]]] = None,
    proof_contract_mode: Optional[str] = None,
) -> str:
    if proof_contract_mode != "phase_checkpoint_v1" or not isinstance(marker_state_bindings, list):
        return ""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for raw in marker_state_bindings:
        if not isinstance(raw, dict):
            continue
        try:
            checkpoint_id = int(raw.get("checkpoint_id"))
        except (TypeError, ValueError):
            continue
        grouped.setdefault(checkpoint_id, []).append(raw)
    if not grouped:
        return ""
    lines = [
        f"{CORE600_CHECKPOINT_PROTOTYPE} {{",
        "    switch(checkpoint_id) {",
    ]
    for checkpoint_id in sorted(grouped):
        lines.append(f"    case {checkpoint_id}u:")
        for binding in grouped[checkpoint_id]:
            token = str(binding.get("marker") or "")
            address = str(binding.get("address") or binding.get("monitor_address") or "0x0")
            mask = str(binding.get("expected_mask") or binding.get("mask") or "0x0")
            relation = str(binding.get("expected_relation") or "")
            expected = "0u" if relation in {"bit_clear", "zero"} else "1u"
            lines.append(
                f'        trace_state_expect("{token}", (uintptr_t){address}, (uint32_t){mask}, {expected});'
            )
        lines.append("        break;")
    lines.extend(
        [
            "    default:",
            "        break;",
            "    }",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


def _marker_state_helper_source(
    marker_state_bindings: Optional[list[dict[str, Any]]] = None,
    proof_contract_mode: Optional[str] = None,
) -> str:
    checkpoint_helper = _checkpoint_helper_source(marker_state_bindings, proof_contract_mode)
    return r"""
// --- CORE600 MARKER-STATE SELF-READBACK HELPER ---
static void core600_marker_putc(char c) {
    while(!(USART2->SR & (1u << 7))) {}
    USART2->DR = (uint32_t)(uint8_t)c;
}
static void core600_marker_puts(const char *s) {
    for(const char *p = s; *p; ++p) {
        core600_marker_putc(*p);
    }
}
static void core600_marker_hex32(uint32_t value) {
    static const char digits[] = "0123456789ABCDEF";
    for(int shift = 28; shift >= 0; shift -= 4) {
        core600_marker_putc(digits[(value >> shift) & 0xFu]);
    }
}
static void core600_marker_newline(void) {
    core600_marker_putc('\r');
    core600_marker_putc('\n');
}
void trace_state_expect(const char *token, uintptr_t address, uint32_t mask, uint32_t expected_nonzero) {
    uint32_t value = *(volatile uint32_t *)address;
    core600_marker_puts("CORE600_MARKER_STATE ");
    core600_marker_puts(token);
    core600_marker_puts(" 0x");
    core600_marker_hex32((uint32_t)address);
    core600_marker_puts(" 0x");
    core600_marker_hex32(mask);
    core600_marker_puts(" 0x");
    core600_marker_hex32(value);
    core600_marker_newline();
    uint32_t masked_nonzero = ((value & mask) != 0u) ? 1u : 0u;
    if(masked_nonzero == (expected_nonzero ? 1u : 0u)) {
        core600_marker_puts("CORE600_CHECK_PASS ");
        core600_marker_puts(token);
        core600_marker_newline();
    }
}
void trace_state(const char *token, uintptr_t address, uint32_t mask) {
    trace_state_expect(token, address, mask, 1u);
}
void trace(const char *token) {
    core600_marker_puts(token);
    core600_marker_newline();
}
""" + checkpoint_helper


def build_compilation_unit(
    requirement: str,
    code: str,
    usart_force: bool = False,
    marker_state_bindings: Optional[list[dict[str, Any]]] = None,
    proof_contract_mode: Optional[str] = None,
) -> str:
    """Build the actual translation unit that GCC sees."""
    context_hints = f"{requirement}\n{code}"
    hw_context = _build_hw_context(context_hints, usart_force=usart_force)
    clean_code = _strip_redefinitions(code, hw_context) if hw_context else code
    if not hw_context:
        return clean_code
    marker_helper = (
        _marker_state_helper_source(marker_state_bindings, proof_contract_mode)
        if usart_force
        else ""
    )
    return f"// --- PRE-INCLUDED HARDWARE CONTEXT ---\n{hw_context}\n{marker_helper}\n// --- GENERATED CODE ---\n{clean_code}"


@dataclass
class CaseResult:
    case_id: int
    level: str
    family_id: str
    cluster_id: str
    requirement: str
    gen_ok: bool = False
    gen_time_ms: float = 0.0
    compile_ok: bool = False
    compile_time_ms: float = 0.0
    compile_errors: int = 0
    compile_warnings: int = 0
    sim_ok: bool = False
    sim_time_ms: float = 0.0
    sim_status: str = ""
    error_category: str = ""
    error_reason: str = ""
    infra_invalid: bool = False
    timestamp: str = ""
    sim_verdict: Optional[dict[str, Any]] = None
    oracle_contract_mode: str = ""
    source_draft_case_id: str = ""
    target_eval_surface_status: str = ""
    promotion_candidate: Optional[bool] = None
    scored_truth: Optional[bool] = None
    eval_set_mutation_allowed: Optional[bool] = None
    generation_source: str = "llm"
    model_generation_started: Optional[bool] = None


class BenchmarkPreflightError(RuntimeError):
    """Raised when benchmark infrastructure is unavailable."""

    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__(f"benchmark_preflight_failed: {';'.join(reasons)}")


def run_benchmark_preflight(skip_sim: bool = False) -> list[str]:
    """Return blocking infrastructure issues that invalidate the run."""
    reasons: list[str] = []
    if not GCC_BIN.exists():
        reasons.append("compiler_missing")
    if not skip_sim:
        if not RENODE_BIN.exists():
            reasons.append("renode_not_found")
        if not RENODE_PLATFORM.exists():
            reasons.append("renode_platform_missing")
    return reasons


def _extract_code(raw: str) -> str:
    """Extract C code from LLM response, stripping markdown fences."""
    text = raw.strip()
    # ```c ... ``` or ``` ... ```
    m = re.search(r"```(?:c|C|cpp)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # If it starts with #include, take it as-is
    if text.startswith("#include"):
        return text
    # Otherwise find first #include to end
    idx = text.find("#include")
    if idx >= 0:
        return text[idx:].strip()
    return text


def _extract_code_from_reasoning(reasoning: str) -> str:
    """Try to extract C code from reasoning_content when content is empty.

    Some reasoning models (e.g. GLM-5.1) may only produce reasoning and leave
    content=None when max_tokens is insufficient. The reasoning may contain
    code blocks with the actual implementation.
    """
    return _extract_code(reasoning)


def _strip_c_comments_for_marker_scan(code: str) -> str:
    without_block_comments = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    return re.sub(r"//.*", "", without_block_comments)


def _missing_required_trace_markers(
    code: str,
    required_ordered_logs: Optional[list[str]] = None,
    proof_contract_mode: Optional[str] = None,
) -> list[str]:
    if not required_ordered_logs:
        return []
    if proof_contract_mode == "phase_checkpoint_v1":
        return []
    source = _strip_c_comments_for_marker_scan(code)
    trace_markers = set(re.findall(r'\btrace\s*\(\s*"([^"]+)"\s*\)', source))
    trace_markers.update(re.findall(r'\btrace_state\s*\(\s*"([^"]+)"\s*,', source))
    trace_markers.update(re.findall(r'\btrace_state_expect\s*\(\s*"([^"]+)"\s*,', source))
    required_markers = [str(token) for token in required_ordered_logs if str(token)]
    return [token for token in required_markers if token not in trace_markers]


def _find_function_body(source: str, function_name: str) -> str | None:
    match = re.search(rf"\b{re.escape(function_name)}\s*\([^)]*\)\s*\{{", source)
    if not match:
        return None
    start = match.end() - 1
    depth = 0
    for index in range(start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1:index]
    return None


def _source_without_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", lambda match: " " * len(match.group(0)), source, flags=re.DOTALL)
    return re.sub(r"//.*", "", source)


def _source_static_gate_errors(
    code: str,
    required_ordered_logs: Optional[list[str]],
    source_static_gate: Optional[dict[str, Any]] = None,
) -> list[str]:
    if not required_ordered_logs or not isinstance(source_static_gate, dict):
        return []
    if source_static_gate.get("mode") != "mainline_ordered_logs_v1":
        return []
    source = _source_without_comments(code)
    main_body = _find_function_body(source, "main")
    if main_body is None:
        return ["source_static_gate_missing_main"]
    infinite_loop_match = re.search(r"\bwhile\s*\(\s*1\s*\)|\bfor\s*\(\s*;\s*;\s*\)", main_body)
    effective_body = main_body[: infinite_loop_match.start()] if infinite_loop_match else main_body
    positions: list[tuple[str, int]] = []
    errors: list[str] = []
    for token in required_ordered_logs:
        trace_pattern = rf"\btrace\s*\(\s*\"{re.escape(str(token))}\"\s*\)"
        trace_state_pattern = rf"\btrace_state\s*\(\s*\"{re.escape(str(token))}\"\s*,"
        trace_state_expect_pattern = rf"\btrace_state_expect\s*\(\s*\"{re.escape(str(token))}\"\s*,"
        match = re.search(trace_pattern, effective_body)
        if not match:
            match = re.search(trace_state_pattern, effective_body)
        if not match:
            match = re.search(trace_state_expect_pattern, effective_body)
        if match:
            positions.append((str(token), match.start()))
            continue
        if (
            re.search(trace_pattern, source)
            or re.search(trace_state_pattern, source)
            or re.search(trace_state_expect_pattern, source)
        ):
            errors.append(f"source_static_gate_trace_not_in_mainline:{token}")
        else:
            errors.append(f"source_static_gate_missing_trace:{token}")
    if errors:
        return errors
    ordered_positions = [position for _token, position in positions]
    if ordered_positions != sorted(ordered_positions):
        ordered = ",".join(token for token, _position in positions)
        errors.append(f"source_static_gate_order_violation:{ordered}")
    if source_static_gate.get("required_prefix_patterns_before_first_marker"):
        first_marker_position = min(ordered_positions)
        prefix_before_first_marker = effective_body[:first_marker_position]
        patterns = source_static_gate.get("required_prefix_patterns_before_first_marker")
        if isinstance(patterns, list):
            for item in patterns:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "unnamed")
                pattern = str(item.get("pattern") or "")
                if pattern and not re.search(pattern, prefix_before_first_marker):
                    errors.append(f"source_static_gate_missing_prefix_pattern:{name}")
    marker_prefix_patterns = source_static_gate.get("required_patterns_before_markers")
    if isinstance(marker_prefix_patterns, list):
        position_by_token = {token: position for token, position in positions}
        for item in marker_prefix_patterns:
            if not isinstance(item, dict):
                continue
            marker = str(item.get("marker") or "")
            name = str(item.get("name") or "unnamed")
            pattern = str(item.get("pattern") or "")
            marker_position = position_by_token.get(marker)
            if pattern and marker_position is not None and not re.search(pattern, effective_body[:marker_position]):
                errors.append(f"source_static_gate_missing_marker_prefix_pattern:{marker}:{name}")
    between_marker_patterns = source_static_gate.get("required_patterns_between_markers")
    if isinstance(between_marker_patterns, list):
        position_by_token = {token: position for token, position in positions}
        for item in between_marker_patterns:
            if not isinstance(item, dict):
                continue
            after_marker = str(item.get("after_marker") or "")
            before_marker = str(item.get("before_marker") or "")
            name = str(item.get("name") or "unnamed")
            pattern = str(item.get("pattern") or "")
            after_position = position_by_token.get(after_marker)
            before_position = position_by_token.get(before_marker)
            if after_position is None:
                errors.append(f"source_static_gate_missing_between_after_marker:{after_marker}:{name}")
                continue
            if before_position is None:
                errors.append(f"source_static_gate_missing_between_before_marker:{before_marker}:{name}")
                continue
            if after_position >= before_position:
                errors.append(f"source_static_gate_invalid_between_marker_order:{after_marker}:{before_marker}:{name}")
                continue
            window = effective_body[after_position:before_position]
            if pattern and not re.search(pattern, window):
                errors.append(f"source_static_gate_missing_between_marker_pattern:{after_marker}:{before_marker}:{name}")
    forbidden_identifiers = source_static_gate.get("forbid_identifiers")
    if isinstance(forbidden_identifiers, list):
        for identifier in forbidden_identifiers:
            name = str(identifier).strip()
            if name and re.search(rf"\b{re.escape(name)}\b", source):
                errors.append(f"source_static_gate_forbidden_identifier:{name}")
    forbidden_patterns = source_static_gate.get("forbid_patterns")
    if isinstance(forbidden_patterns, list):
        for item in forbidden_patterns:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "unnamed")
            pattern = str(item.get("pattern") or "")
            if pattern and re.search(pattern, source):
                errors.append(f"source_static_gate_forbidden_pattern:{name}")
    if source_static_gate.get("no_wait_loops_before_required_markers"):
        last_marker_position = max(ordered_positions)
        prefix = effective_body[:last_marker_position]
        wait_match = re.search(r"\b(?:while|for)\s*\(", prefix)
        if wait_match:
            errors.append("source_static_gate_wait_loop_before_required_markers")
    if source_static_gate.get("forbid_exti_swier_before_required_markers"):
        last_marker_position = max(ordered_positions)
        prefix = effective_body[:last_marker_position]
        if re.search(r"\bEXTI\s*->\s*SWIER\b", prefix):
            errors.append("source_static_gate_exti_swier_before_required_markers")
    if source_static_gate.get("forbid_irq_enable_before_required_markers"):
        last_marker_position = max(ordered_positions)
        prefix = effective_body[:last_marker_position]
        if re.search(r"\b__enable_irq\s*\(", prefix):
            errors.append("source_static_gate_irq_enable_before_required_markers")
    forbidden_after_final_marker = source_static_gate.get("forbid_patterns_after_final_marker")
    if isinstance(forbidden_after_final_marker, list):
        last_marker_position = max(ordered_positions)
        suffix = main_body[last_marker_position:]
        for item in forbidden_after_final_marker:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "unnamed")
            pattern = str(item.get("pattern") or "")
            if pattern and re.search(pattern, suffix):
                errors.append(f"source_static_gate_forbidden_pattern_after_final_marker:{name}")
    return errors


def _validate_code(
    code: str,
    required_ordered_logs: Optional[list[str]] = None,
    source_static_gate: Optional[dict[str, Any]] = None,
    proof_contract_mode: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    errors = []
    if not code or len(code) < 50:
        errors.append("code_too_short")
        return False, errors
    if "main" not in code:
        errors.append("missing_main")
        return False, errors
    if "..." in code and code.count("...") > 2:
        errors.append("placeholder_detected")
    # Check for obviously truncated code: no closing brace for main
    if not code.rstrip().endswith("}"):
        errors.append("truncated_code")
        return False, errors
    # Check for TODO/FIXME placeholders
    if re.search(r'\b(TODO|FIXME|XXX|HACK)\b', code, re.IGNORECASE):
        errors.append("todo_fixme_in_code")
        return False, errors
    missing_markers = _missing_required_trace_markers(
        code,
        required_ordered_logs,
        proof_contract_mode=proof_contract_mode,
    )
    if missing_markers:
        errors.append(f"missing_required_trace_markers:{','.join(missing_markers)}")
        return False, errors
    source_gate_errors = _source_static_gate_errors(
        code,
        required_ordered_logs=required_ordered_logs,
        source_static_gate=source_static_gate,
    )
    if source_gate_errors:
        return False, source_gate_errors
    return True, errors


def _source_gate_repair_budget(source_static_gate: Optional[dict[str, Any]]) -> int:
    if not isinstance(source_static_gate, dict):
        return 0
    budget = 2 if source_static_gate.get("strict_timeout_safe_template") else 1
    configured = source_static_gate.get("source_gate_repair_budget")
    if isinstance(configured, int) and not isinstance(configured, bool) and configured >= 0:
        budget = min(configured, 5)
    return budget


def _source_static_gate_pattern_example(name: str, pattern: str) -> str:
    lowered = name.lower()
    if ("clear" in lowered or "disable" in lowered or "deassert" in lowered) and ("gpioa" in lowered or "gpio" in lowered):
        return "GPIOA->ODR &= ~0x20u;"
    if ("clear" in lowered or "disable" in lowered or "deassert" in lowered) and "tim2" in lowered:
        return "TIM2->CR1 &= ~0x1u;"
    if "tim2" in lowered:
        return "TIM2->CR1 |= 0x1u;"
    if "usart2" in lowered and "cr1" in lowered:
        return "USART2->CR1 |= (1u << 3);"
    if "gpioa" in lowered or "gpio" in lowered:
        return "GPIOA->ODR |= 0x20u;"
    if "exti" in lowered and ("mask" in lowered or "imr" in lowered):
        return "EXTI->IMR |= 0x1u;"
    if "exti" in lowered and ("edge" in lowered or "rtsr" in lowered):
        return "EXTI->RTSR |= 0x1u;"
    if "flash" in lowered or "acr" in lowered:
        return "FLASH->ACR |= 0x7u;"
    if "i2c" in lowered and "cr1" in lowered:
        return "I2C1->CR1 |= 0x0001u;"
    if "spi2" in lowered and "rcc" in lowered:
        return "RCC->APB1ENR |= (1u << 14);"
    if "spi2" in lowered and "cr1" in lowered:
        return "SPI2->CR1 |= (1u << 6);"
    if "spi2" in lowered and "cr2" in lowered:
        return "SPI2->CR2 |= 0x1u;"
    return ""


def _source_static_gate_repair_guidance(source_static_gate: Optional[dict[str, Any]]) -> str:
    if not isinstance(source_static_gate, dict):
        return ""
    guidance: list[str] = []
    prefix_patterns = source_static_gate.get("required_prefix_patterns_before_first_marker")
    if isinstance(prefix_patterns, list):
        required = []
        for item in prefix_patterns:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            pattern = str(item.get("pattern") or "").strip()
            if name and pattern:
                example = _source_static_gate_pattern_example(name, pattern)
                suffix = f" Example accepted statement: {example}" if example else ""
                required.append(f"{name} must match /{pattern}/ before the first required trace marker.{suffix}")
        if required:
            guidance.append("Required prefix patterns: " + "; ".join(required) + ".")
    marker_prefix_patterns = source_static_gate.get("required_patterns_before_markers")
    if isinstance(marker_prefix_patterns, list):
        required_by_marker = []
        for item in marker_prefix_patterns:
            if not isinstance(item, dict):
                continue
            marker = str(item.get("marker") or "").strip()
            name = str(item.get("name") or "").strip()
            pattern = str(item.get("pattern") or "").strip()
            if marker and name and pattern:
                example = _source_static_gate_pattern_example(name, pattern)
                suffix = f" Example accepted statement: {example}" if example else ""
                required_by_marker.append(f"{name} must match /{pattern}/ before trace(\"{marker}\").{suffix}")
        if required_by_marker:
            guidance.append("Required staged marker-prefix patterns: " + "; ".join(required_by_marker) + ".")
    between_marker_patterns = source_static_gate.get("required_patterns_between_markers")
    if isinstance(between_marker_patterns, list):
        required_between = []
        for item in between_marker_patterns:
            if not isinstance(item, dict):
                continue
            after_marker = str(item.get("after_marker") or "").strip()
            before_marker = str(item.get("before_marker") or "").strip()
            name = str(item.get("name") or "").strip()
            pattern = str(item.get("pattern") or "").strip()
            if after_marker and before_marker and name and pattern:
                example = _source_static_gate_pattern_example(name, pattern)
                suffix = f" Example accepted statement: {example}" if example else ""
                required_between.append(
                    f"{name} must match /{pattern}/ between trace(\"{after_marker}\") and trace(\"{before_marker}\").{suffix}"
                )
        if required_between:
            guidance.append("Required between-marker patterns: " + "; ".join(required_between) + ".")
    forbidden_after = source_static_gate.get("forbid_patterns_after_final_marker")
    if isinstance(forbidden_after, list):
        forbidden = []
        for item in forbidden_after:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            pattern = str(item.get("pattern") or "").strip()
            if name and pattern:
                forbidden.append(f"{name} must not match /{pattern}/ after the final required trace marker")
        if forbidden:
            guidance.append("Forbidden after final marker: " + "; ".join(forbidden) + ".")
    forbidden_patterns = source_static_gate.get("forbid_patterns")
    if isinstance(forbidden_patterns, list):
        forbidden_anywhere = []
        for item in forbidden_patterns:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            pattern = str(item.get("pattern") or "").strip()
            if name and pattern:
                forbidden_anywhere.append(f"{name} must not match /{pattern}/ anywhere in generated source")
        if forbidden_anywhere:
            guidance.append("Forbidden source patterns: " + "; ".join(forbidden_anywhere) + ".")
    if source_static_gate.get("no_wait_loops_before_required_markers"):
        guidance.append("No while(...) or for(...) loop may appear before the final required trace marker.")
    if source_static_gate.get("forbid_exti_swier_before_required_markers"):
        guidance.append("Do not write EXTI->SWIER before all required trace markers.")
    if source_static_gate.get("forbid_irq_enable_before_required_markers"):
        guidance.append("Do not call __enable_irq() before all required trace markers.")
    return " ".join(guidance)


def call_siliconflow(
    model: str,
    requirement: str,
    api_key: str,
    max_retries: int = LLM_MAX_RETRIES,
    timeout: int = LLM_TIMEOUT,
    required_ordered_logs: Optional[list[str]] = None,
    marker_state_bindings: Optional[list[dict[str, Any]]] = None,
    proof_contract_mode: Optional[str] = None,
    source_static_gate: Optional[dict[str, Any]] = None,
    api_url: Optional[str] = None,
) -> Tuple[bool, str, float, List[str]]:
    """Call LLM API for code generation. Returns (ok, code, time_ms, errors)."""
    url = api_url or SF_API_URL
    system_prompt = _build_system_prompt(
        requirement,
        required_ordered_logs=required_ordered_logs,
        marker_state_bindings=marker_state_bindings,
        proof_contract_mode=proof_contract_mode,
    )
    source_gate_guidance = _source_static_gate_repair_guidance(source_static_gate)
    initial_user_content = "Provide complete C code implementing the task. Respond with code only. Start your response EXACTLY with: #include <stdint.h>"
    if source_gate_guidance:
        initial_user_content = (
            "Source-static gate constraints for this case: "
            f"{source_gate_guidance} "
            + initial_user_content
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_user_content},
    ]
    errors = []
    start = time.time()

    marker_repair_budget = 1 if required_ordered_logs else 0
    source_gate_repair_budget = _source_gate_repair_budget(source_static_gate)
    total_attempts = max(1, max_retries) + marker_repair_budget + source_gate_repair_budget
    for attempt in range(total_attempts):
        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": MAX_TOKENS,
                    "temperature": float(os.getenv("CORE600_TEMPERATURE", "0.0")),
                },
                timeout=timeout,
            )
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                errors.append(f"rate_limit_429, wait {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                errors.append(f"http_{resp.status_code}: {resp.text[:200]}")
                if resp.status_code in HTTP_FAIL_FAST_STATUS_CODES:
                    break
                time.sleep(2)
                continue

            data = resp.json()
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                errors.append("empty_choices")
                continue
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            msg = first_choice.get("message", {})
            if not isinstance(msg, dict):
                errors.append("empty_message")
                continue
            raw_output = msg.get("content") or ""
            # Fallback for reasoning models: if content is None/empty but
            # reasoning_content exists with embedded code, try to extract from it
            if not raw_output:
                reasoning = msg.get("reasoning_content", "")
                if reasoning:
                    raw_output = _extract_code_from_reasoning(reasoning)
            if not raw_output:
                errors.append("empty_response")
                continue

            code = _extract_code(raw_output)
            valid, val_errors = _validate_code(
                code,
                required_ordered_logs=required_ordered_logs,
                source_static_gate=source_static_gate,
                proof_contract_mode=proof_contract_mode,
            )
            elapsed = (time.time() - start) * 1000
            if not valid:
                errors.extend(val_errors)
                marker_errors = [
                    error for error in val_errors if error.startswith("missing_required_trace_markers:")
                ]
                if marker_errors and marker_repair_budget > 0:
                    marker_repair_budget -= 1
                    missing = marker_errors[-1].split(":", 1)[1]
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous code missed exact required helper markers: "
                                f"{missing}. Return a complete corrected C file only. Each listed marker must "
                                "appear as trace(\"TOKEN\"), trace_state(\"TOKEN\", ...), or "
                                "trace_state_expect(\"TOKEN\", ...) in source, not only in comments, and execute in order."
                            ),
                        }
                    )
                    continue
                source_gate_errors = [
                    error for error in val_errors if error.startswith("source_static_gate_")
                ]
                if source_gate_errors and source_gate_repair_budget > 0:
                    source_gate_repair_budget -= 1
                    first_required_marker = str(required_ordered_logs[0]) if required_ordered_logs else "the first required marker"
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous code failed the source-static mainline gate: "
                                f"{'; '.join(source_gate_errors)}. Return a complete corrected C file only. "
                                "Every required trace/helper marker must appear in main(), before any final "
                                "while(1) or for(;;), in exact required order. Do not place required markers only "
                                "inside interrupt handlers, callbacks, branches, or wait loops. Do not use while/for "
                                "wait loops before the final required trace marker. For timeout-safe cases, do not write "
                                "EXTI->SWIER or call __enable_irq() before all required trace markers have been emitted. "
                                f"For any source_static_gate_missing_prefix_pattern error, move the matching concrete "
                                f"register write above the first required trace marker trace(\"{first_required_marker}\"); "
                                "a write immediately below that first trace marker is still a failure. "
                                f"{_source_static_gate_repair_guidance(source_static_gate)}"
                            ),
                        }
                    )
                    continue
                return False, code, elapsed, errors
            return True, code, elapsed, []

        except requests.Timeout:
            errors.append(f"timeout_attempt_{attempt+1}")
        except Exception as e:
            errors.append(f"exception_attempt_{attempt+1}: {str(e)[:150]}")
            time.sleep(1)

    elapsed = (time.time() - start) * 1000
    return False, "", elapsed, errors


def _compact_llm_error_reason(errors: list[str]) -> str:
    text = ";".join(errors[:3])
    lower = text.lower()
    if "http_403" in lower and "model is private" in lower:
        return "http_403_model_is_private"
    if "http_401" in lower:
        return "http_401_unauthorized"
    if "http_403" in lower:
        return "http_403_forbidden"
    if "http_404" in lower:
        return "http_404_model_or_endpoint_not_found"
    if "timeout_attempt" in lower:
        return "llm_timeout"
    if "rate_limit_429" in lower:
        return "rate_limit_429"
    if "empty_choices" in lower:
        return "llm_empty_choices"
    if "empty_message" in lower:
        return "llm_empty_message"
    if "empty_response" in lower:
        return "llm_empty_response"
    if text:
        return text[:200]
    return "unknown_llm_error"


def _compact_compile_error_reason(errors: list[str]) -> str:
    for raw in errors:
        text = str(raw).strip()
        if text in INFRA_INVALID_REASONS:
            return text

    for raw in errors:
        for line in str(raw).splitlines():
            normalized = re.sub(r"\s+", " ", line.strip())
            normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
            match = re.search(r"\b(?:fatal\s+)?error:\s*(.+)", normalized, flags=re.IGNORECASE)
            if not match:
                continue
            reason = match.group(1).strip()
            undeclared = re.search(r"'([^']+)'\s+undeclared", reason)
            if undeclared:
                return f"{undeclared.group(1)} undeclared"
            return reason[:200]

    for raw in errors:
        text = re.sub(r"\s+", " ", str(raw).strip())
        if not text:
            continue
        if re.search(r"[/\\]", text) and "error:" not in text.lower():
            continue
        return text[:200]

    return "unknown_compile_error"


_ALL_KNOWN_STRUCTS = {
    'RCC_TypeDef', 'GPIO_TypeDef', 'NVIC_TypeDef', 'SCB_TypeDef',
    'STK_TypeDef', 'FLASH_TypeDef', 'EXTI_TypeDef', 'SYSCFG_TypeDef',
    'USART_TypeDef', 'TIM_TypeDef', 'TIM_Advanced_TypeDef',
    'DMA_TypeDef', 'DMA_Stream_TypeDef',
    'ADC_TypeDef', 'ADC_Common_TypeDef',
    'CAN_TypeDef', 'CAN_TxMailBox_TypeDef', 'CAN_FIFOMailBox_TypeDef', 'CAN_FilterRegister_TypeDef',
    'SPI_TypeDef', 'I2C_TypeDef',
    'WWDG_TypeDef', 'IWDG_TypeDef',
    'CRC_TypeDef', 'DAC_TypeDef',
}


def _strip_redefinitions(code: str, hw_context: str) -> str:
    """Remove struct/enum/macro/inline-func redefinitions from model code that already exist in hw_context."""
    # Collect struct names, enum names, #define macros, and inline functions from hw_context
    defined_structs = set(re.findall(r'typedef\s+struct.*?\}\s*(\w+)\s*;', hw_context, re.DOTALL))
    defined_enums = set(re.findall(r'typedef\s+enum.*?\}\s*(\w+)\s*;', hw_context, re.DOTALL))
    defined_macros = set(re.findall(r'#define\s+(\w+)', hw_context))
    defined_inline_funcs = set(re.findall(r'static\s+inline\s+\w[\w\s\*]*?\b(\w+)\s*\(', hw_context))

    # Inline function names should also be stripped if re-#defined
    defined_macros |= defined_inline_funcs

    result = code

    # Remove typedef struct redefinitions (match from typedef to "} STRUCTNAME;")
    # Use union with _ALL_KNOWN_STRUCTS to cover structs defined in other slices
    for sname in sorted(defined_structs | _ALL_KNOWN_STRUCTS):
        pattern = r'typedef\s+struct\b[\s\S]*?\}\s*' + re.escape(sname) + r'\s*;\s*\n?'
        result = re.sub(pattern, '', result)

    # Remove typedef enum redefinitions
    for ename in sorted(defined_enums):
        pattern = r'typedef\s+enum\b[\s\S]*?\}\s*' + re.escape(ename) + r'\s*;\s*\n?'
        result = re.sub(pattern, '', result)

    # Remove #define redefinitions (including multi-line with backslash continuation)
    for mname in sorted(defined_macros):
        pattern = r'#define\s+' + re.escape(mname) + r'\b[^\n\\]*(?:\\\n[^\n\\]*)*(?:\\\n)?\s*\n?'
        result = re.sub(pattern, '', result)

    # Remove static inline / inline function redefinitions
    for fname in sorted(defined_inline_funcs):
        pattern = (
            r'(?:static\s+)?(?:inline\s+)?'
            r'\w[\w\s\*]*?\b'
            + re.escape(fname)
            + r'\s*\([^)]*\)\s*'
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            r'\s*\n?'
        )
        result = re.sub(pattern, '', result)

    # The runner owns these helper functions.  A model-provided extern with a
    # slightly different prototype is a deterministic scaffold mismatch, not
    # task code.  Remove declarations while still rejecting actual definitions.
    runner_helper_names = [
        "CORE600_CHECKPOINT",
        "trace_state_expect",
        "trace_state",
        "trace",
        "CORE600_CHECK_PASS",
    ]
    for helper_name in runner_helper_names:
        pattern = (
            r"^\s*(?:extern\s+)?void\s+"
            + re.escape(helper_name)
            + r"\s*\([^;{}]*\)\s*;\s*\n?"
        )
        result = re.sub(pattern, "", result, flags=re.MULTILINE)

    return result


def compile_code(
    code: str,
    requirement: str,
    test_id: int,
    build_dir: Path,
    compile_log_dir: Path,
    elf_output_dir: Path,
    usart_force: bool = False,
    marker_state_bindings: Optional[list[dict[str, Any]]] = None,
    proof_contract_mode: Optional[str] = None,
    compile_timeout: int = COMPILE_TIMEOUT,
) -> Tuple[bool, float, int, int, List[str], Optional[Path]]:
    """Compile C code with ARM GCC. Returns (ok, time_ms, warnings, errors, msgs, elf_path)."""
    build_dir.mkdir(parents=True, exist_ok=True)
    compile_log_dir.mkdir(parents=True, exist_ok=True)
    elf_output_dir.mkdir(parents=True, exist_ok=True)

    source = build_dir / f"core600_{test_id}.c"
    source.write_text(
        build_compilation_unit(
            requirement,
            code,
            usart_force=usart_force,
            marker_state_bindings=marker_state_bindings,
            proof_contract_mode=proof_contract_mode,
        )
    )

    flags = [
        "-mcpu=cortex-m4", "-mthumb", "-ffreestanding",
        "-nostdlib", "-nostartfiles", "-O0", "-g",
        "-Wall", "-Wextra",
    ]
    elf_file = elf_output_dir / f"core600_{test_id}.elf"
    log_file = compile_log_dir / f"core600_{test_id}.log"

    start = time.time()
    try:
        cmd = [str(GCC_BIN)] + flags + [
            "-T", str(TEMPLATE_LINKER),
            str(TEMPLATE_STARTUP),
            str(source),
            "-o", str(elf_file),
        ]
        result = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=compile_timeout
        )
        stderr = result.stderr or ""
        log_file.write_text(f"CMD: {' '.join(cmd)}\n---STDERR---\n{stderr}\n")
        warnings = stderr.count("warning:")
        errors = stderr.count("error:")
        ok = result.returncode == 0
        elapsed = (time.time() - start) * 1000
        if not ok:
            return False, elapsed, warnings, errors, [(stderr or "")[:500]], None
        return True, elapsed, warnings, errors, [], elf_file

    except FileNotFoundError:
        log_file.write_text("arm-none-eabi-gcc not found\n")
        return False, 0, 0, 1, ["compiler_missing"], None
    except subprocess.TimeoutExpired:
        log_file.write_text("compile_timeout\n")
        return False, (time.time() - start) * 1000, 0, 1, ["compile_timeout"], None
    except Exception as e:
        log_file.write_text(f"compile_exception: {e}\n")
        return False, (time.time() - start) * 1000, 0, 1, [str(e)[:200]], None


def _process_stream_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _trim_log_for_evidence(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    marker = (
        "\n---CORE600_LOG_TRUNCATED_PRESERVED_HEAD_TAIL---\n"
        f"original_chars={len(text)} limit={limit}\n"
        "---CORE600_LOG_TAIL---\n"
    )
    head_len = max(0, limit // 2)
    tail_len = max(0, limit - head_len)
    return text[:head_len] + marker + text[-tail_len:]


def _write_renode_log(
    renode_log_dir: Path,
    test_id: int,
    stdout: Any,
    stderr: Any,
    timeout_seconds: Optional[int] = None,
) -> Path:
    log_file = renode_log_dir / f"core600_{test_id}.log"
    stdout_text = _process_stream_text(stdout)
    stderr_text = _process_stream_text(stderr)
    prefix = ""
    if timeout_seconds is not None:
        prefix = f"---RENODE TIMEOUT---\ntimeout_seconds={timeout_seconds}\n"
    combined = prefix + stdout_text + "\n---STDERR---\n" + stderr_text
    log_file.write_text(_trim_log_for_evidence(combined, 50000), encoding="utf-8")
    return log_file


def _core600_renode_platform(renode_log_dir: Path) -> Path:
    """Return a Core600 platform copy that avoids Renode SVD init stalls.

    Readback parsing no longer depends on SVD register names; it accepts raw
    ReadDoubleWord values. Skipping ApplySVD avoids batch runs hanging before
    ELF load. Set CORE600_RENODE_APPLY_SVD=1 to force the repo-local SVD.
    """
    apply_svd = os.getenv("CORE600_RENODE_APPLY_SVD", "0") == "1"
    if apply_svd and not LOCAL_STM32F40X_SVD.exists():
        return RENODE_PLATFORM
    renode_log_dir.mkdir(parents=True, exist_ok=True)
    platform_copy = renode_log_dir / "core600_stm32f4_local_svd.repl"
    source = RENODE_PLATFORM.read_text(encoding="utf-8")
    if apply_svd:
        patched = source.replace(
            "ApplySVD @https://dl.antmicro.com/projects/renode/svd/STM32F40x.svd.gz",
            f"ApplySVD @{LOCAL_STM32F40X_SVD}",
        )
    else:
        patched = source.replace(
            "        ApplySVD @https://dl.antmicro.com/projects/renode/svd/STM32F40x.svd.gz",
            "        // Core600 runner disables ApplySVD by default to avoid Renode init stalls",
        )
    if not platform_copy.exists() or platform_copy.read_text(encoding="utf-8") != patched:
        platform_copy.write_text(patched, encoding="utf-8")
    return platform_copy


def simulate_elf(
    elf_file: Path,
    test_id: int,
    renode_log_dir: Path,
    sim_seconds: float = SIM_SECONDS,
    case: Optional[dict[str, Any]] = None,
    semantic_oracle: bool = False,
    renode_timeout_floor: int = SIM_TIMEOUT_FLOOR,
    trace_peripheral_access: bool = RENODE_TRACE_PERIPHERAL_ACCESS,
) -> Tuple[bool, float, str, str, str, Optional[dict[str, Any]]]:
    """Run Renode simulation.

    Returns (ok, time_ms, status, error_category, error_reason, sim_verdict).
    When semantic_oracle=True, sim_verdict contains the SemanticVerdict dict
    and sim_ok is influenced by likely_correct.
    """
    renode_log_dir.mkdir(parents=True, exist_ok=True)

    if not RENODE_BIN.exists() or not RENODE_PLATFORM.exists():
        return False, 0, "env_missing", "infra", "renode_not_found", None
    platform_file = _core600_renode_platform(renode_log_dir)

    before_load_stimulus = _renode_stimulus_commands(case, "before_load")
    before_run_stimulus = _renode_stimulus_commands(case, "before_run")
    before_readback_stimulus = _renode_stimulus_commands(case, "before_readback")
    before_readback_internal_events = _renode_internal_event_commands(case, "before_readback")
    after_readback_stimulus = _renode_stimulus_commands(case, "after_readback")
    before_readbacks = _state_readback_commands(case, "BEFORE")
    after_readbacks = _state_readback_commands(case, "AFTER")
    peripheral_trace_command = "sysbus LogAllPeripheralsAccess true\n" if trace_peripheral_access else ""
    script = f'''using sysbus
mach create
machine LoadPlatformDescription @{platform_file}
{peripheral_trace_command}\
showAnalyzer usart1
showAnalyzer usart2
showAnalyzer usart3
showAnalyzer usart6
{before_load_stimulus}\
{before_readbacks}\
sysbus LoadELF @{elf_file}
{before_run_stimulus}\
emulation RunFor "{sim_seconds}s"
{before_readback_stimulus}\
{before_readback_internal_events}\
{after_readbacks}\
{after_readback_stimulus}\
quit
'''
    script_file = renode_log_dir / f"core600_{test_id}.resc"
    script_file.write_text(script)

    start = time.time()
    renode_timeout = max(renode_timeout_floor, sim_seconds + 10)
    try:
        result = subprocess.run(
            [str(RENODE_BIN), "--disable-xwt", "--console", str(script_file)],
            capture_output=True, text=True, timeout=renode_timeout,
        )
        elapsed = (time.time() - start) * 1000
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        combined = stdout + "\n---STDERR---\n" + stderr

        _write_renode_log(renode_log_dir, test_id, stdout, stderr)

        if "HardFault" in stdout or "HardFault" in stderr:
            return False, elapsed, "hardfault", "code", "hardfault", None
        if result.returncode != 0:
            return False, elapsed, "renode_error", "infra", "renode_nonzero_exit", None
        has_write = _has_observable_peripheral_activity(combined, case=case)
        if not has_write:
            return False, elapsed, "no_peripheral_write", "code", "no_peripheral_write", None

        # Semantic oracle analysis (always compute, store in verdict)
        verdict = _semantic_oracle(combined, case)
        verdict_dict = asdict(verdict)

        if _requires_semantic_gating(case, semantic_oracle) and not verdict.likely_correct:
            return False, elapsed, "semantic_fail", "code", "semantic_oracle_rejection", verdict_dict

        return True, elapsed, "ok", "", "", verdict_dict

    except subprocess.TimeoutExpired as e:
        stdout = getattr(e, "stdout", None)
        if stdout is None:
            stdout = getattr(e, "output", None)
        stderr = getattr(e, "stderr", None)
        stdout_text = _process_stream_text(stdout)
        stderr_text = _process_stream_text(stderr)
        combined = stdout_text + "\n---STDERR---\n" + stderr_text
        _write_renode_log(
            renode_log_dir,
            test_id,
            stdout,
            stderr,
            timeout_seconds=renode_timeout,
        )
        if _has_observable_peripheral_activity(combined, case=case):
            verdict = _semantic_oracle(combined, case)
            verdict_dict = asdict(verdict)
            if _requires_semantic_gating(case, semantic_oracle):
                if verdict.likely_correct:
                    return True, (time.time() - start) * 1000, "ok", "", "", verdict_dict
                if (
                    "CORE600_READBACK_AFTER" in combined
                    or _has_required_ordered_logs(_extract_usart_terminal_output(combined) or combined, case)
                ):
                    return (
                        False,
                        (time.time() - start) * 1000,
                        "semantic_fail",
                        "code",
                        "semantic_oracle_rejection_after_timeout",
                        verdict_dict,
                    )
        return False, (time.time() - start) * 1000, "sim_timeout", "env", "sim_timeout", None
    except Exception as e:
        return False, (time.time() - start) * 1000, "sim_exception", "infra", "sim_exception", None


def _extract_usart_terminal_output(log_text: str) -> str:
    """Extract USART terminal output from Renode log after showAnalyzer is enabled.

    Renode's UART analyzer (LoggingUartAnalyzer) emits lines like:
        HH:MM:SS.mmmm [INFO] usart2: [host: Xs (+Ys)|virt: As (+Bs)] firmware text
    We extract only the firmware text after the timestamp bracket, skipping
    Renode WARNING lines that also start with `usartN:` but are structured differently:
        HH:MM:SS.mmmm [WARNING] usart2: Unhandled write to offset ...
    """
    output_parts: list[str] = []
    for match in re.finditer(
        r"\[INFO\]\s*usart\d+\s*:\s*(?:\[.*?\]\s*)?(.+)$",
        log_text,
        flags=re.MULTILINE,
    ):
        line = match.group(1).strip()
        if line and "Unhandled write" not in line and "not supported" not in line:
            output_parts.append(line)
    return "\n".join(output_parts)


# ── Semantic Oracle (Phase 2) ──────────────────────────────────────────

@dataclass
class SemanticVerdict:
    """Result of semantic oracle analysis."""
    usart_tokens_found: list[str]
    register_anchors: list[str]
    svd_tags: list[str]
    forbidden_shortcuts: list[str]
    control_pattern_hits: list[str]
    state_readback_expectations: list[dict[str, Any]]
    state_readback_failures: list[str]
    marker_state_bindings: list[dict[str, Any]]
    marker_state_failures: list[str]
    helper_check_pass_tokens: list[str]
    semantic_score: float  # 0.0-1.0
    likely_correct: bool
    next200_policy_diagnostics: list[dict[str, Any]] = field(default_factory=list)


_STRICT_ORACLE_CONTRACT_MODES = {
    "semantic_strict",
    "semantic_strict_state_readback_anchor_candidate_v1",
}


def _requires_semantic_gating(case: Optional[dict[str, Any]], semantic_oracle: bool = False) -> bool:
    """Return whether sim_ok must be gated by the semantic verdict."""
    if semantic_oracle:
        return True
    if isinstance(case, dict) and case.get("oracle_contract_mode") in _STRICT_ORACLE_CONTRACT_MODES:
        return True
    return False


def _next200_policy_annotations(case: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(case, dict):
        return []
    annotations = case.get("next200_policy_candidate_annotations")
    if not isinstance(annotations, list):
        return []
    return [dict(item) for item in annotations if isinstance(item, dict)]


def _next200_policy_candidate_ids(cases: list[dict[str, Any]]) -> set[str]:
    candidate_ids: set[str] = set()
    for case in cases:
        for annotation in _next200_policy_annotations(case):
            candidate_id = str(annotation.get("candidate_id") or "").strip()
            if candidate_id:
                candidate_ids.add(candidate_id)
    return candidate_ids


def _next200_validated_candidate_annotations(case: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(case, dict):
        return []
    annotations = case.get("next200_validated_candidate_annotations")
    if not isinstance(annotations, list):
        return []
    return [dict(item) for item in annotations if isinstance(item, dict)]


def _next200_validated_candidate_ids(cases: list[dict[str, Any]]) -> set[str]:
    candidate_ids: set[str] = set()
    for case in cases:
        for annotation in _next200_validated_candidate_annotations(case):
            candidate_id = str(annotation.get("candidate_id") or "").strip()
            if candidate_id:
                candidate_ids.add(candidate_id)
    return candidate_ids


def _contains_raw_sysbus_write(value: Any) -> bool:
    if isinstance(value, str):
        return re.search(r"\bsysbus\s+Write(?:DoubleWord|Word|UInt32|UInt16|Byte)\b", value, re.IGNORECASE) is not None
    if isinstance(value, dict):
        return any(_contains_raw_sysbus_write(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_raw_sysbus_write(item) for item in value)
    return False


def validate_next200_policy_support_preflight(
    cases: list[dict[str, Any]],
    run_context: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Validate that next200 annotation support is explicitly hash-bound."""
    context = run_context if isinstance(run_context, dict) else {}
    support_enabled = bool(context.get(NEXT200_POLICY_SUPPORT_FLAG))
    candidate_ids = _next200_policy_candidate_ids(cases)
    blockers: list[str] = []
    if support_enabled:
        stable_sha = str(context.get(NEXT200_POLICY_SURFACE_STABLE_SHA_KEY) or "").strip()
        if not stable_sha:
            blockers.append("missing_candidate_surface_hash")
        elif stable_sha != NEXT200_POLICY_CANDIDATE_SURFACE_STABLE_SHA256:
            blockers.append("candidate_surface_hash_mismatch")
        unknown_ids = sorted(candidate_ids - NEXT200_KNOWN_POLICY_CANDIDATE_IDS)
        if unknown_ids:
            blockers.append("unknown_candidate_id")
        if context.get("rewrite_existing_results") is True or context.get("existing_results_mutation_allowed") is True:
            blockers.append("existing_results_rewrite_disallowed")
    return {
        "support_enabled": support_enabled,
        "valid": not blockers,
        "blockers": blockers,
        "candidate_id_count": len(candidate_ids),
        "unknown_candidate_ids": sorted(candidate_ids - NEXT200_KNOWN_POLICY_CANDIDATE_IDS),
    }


def validate_next200_validated_candidate_support_preflight(
    cases: list[dict[str, Any]],
    run_context: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Validate hash-bound local support for next200 validated diagnostic candidates."""
    context = run_context if isinstance(run_context, dict) else {}
    support_enabled = bool(context.get(NEXT200_VALIDATED_SUPPORT_FLAG))
    candidate_ids = _next200_validated_candidate_ids(cases)
    unknown_ids = sorted(candidate_ids - NEXT200_VALIDATED_KNOWN_CANDIDATE_IDS)
    blockers: list[str] = []
    if support_enabled:
        support_sha = str(context.get(NEXT200_VALIDATED_RUNNER_SUPPORT_STABLE_SHA_KEY) or "").strip()
        fixture_sha = str(context.get(NEXT200_VALIDATED_NEGATIVE_FIXTURE_STABLE_SHA_KEY) or "").strip()
        if not support_sha:
            blockers.append("missing_validated_runner_support_hash")
        elif support_sha != NEXT200_VALIDATED_RUNNER_SUPPORT_STABLE_SHA256:
            blockers.append("validated_runner_support_hash_mismatch")
        if not fixture_sha:
            blockers.append("missing_validated_negative_fixture_hash")
        elif fixture_sha != NEXT200_VALIDATED_NEGATIVE_FIXTURE_STABLE_SHA256:
            blockers.append("validated_negative_fixture_hash_mismatch")
        if unknown_ids:
            blockers.append("unknown_validated_candidate_id")
        if _contains_raw_sysbus_write(cases) or _contains_raw_sysbus_write(context.get("renode_stimulus_commands")):
            blockers.append("raw_sysbus_write_disallowed")
        if context.get("rewrite_existing_results") is True or context.get("existing_results_mutation_allowed") is True:
            blockers.append("existing_results_rewrite_disallowed")
        if context.get("runtime_release_requested") is True:
            if context.get("negative_controls_executed") is not True:
                blockers.append("negative_controls_not_executed")
            if context.get("fresh_output_root_declared") is not True:
                blockers.append("fresh_output_root_required")
            if context.get("runtime_release_packet_present") is not True:
                blockers.append("runtime_release_packet_missing")
    return {
        "support_enabled": support_enabled,
        "valid": not blockers,
        "blockers": blockers,
        "candidate_id_count": len(candidate_ids),
        "unknown_candidate_ids": unknown_ids,
        "runtime_release_allowed": support_enabled
        and not blockers
        and context.get("runtime_release_requested") is True
        and context.get("negative_controls_executed") is True
        and context.get("fresh_output_root_declared") is True
        and context.get("runtime_release_packet_present") is True,
    }


def _state_readback_expectations(case: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return row-level state readback expectations from either promoted field."""
    if not isinstance(case, dict):
        return []
    expectations = case.get("state_readback_expectations")
    if not isinstance(expectations, list):
        renode_verification = case.get("renode_verification") or {}
        if isinstance(renode_verification, dict):
            expectations = renode_verification.get("state_readback_expectations")
    if not isinstance(expectations, list):
        return []
    return [dict(item) for item in expectations if isinstance(item, dict)]


def _state_readback_label(index: int, expectation: dict[str, Any], phase: str) -> str:
    target = str(expectation.get("target") or f"STATE_{index}")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", target).strip("_").upper()
    if not slug:
        slug = f"STATE_{index}"
    return f"CORE600_READBACK_{phase.upper()}_{index:03d}_{slug}"


def _state_readback_commands(case: Optional[dict[str, Any]], phase: str) -> str:
    lines: list[str] = []
    for index, expectation in enumerate(_state_readback_expectations(case), start=1):
        command = str(expectation.get("monitor_command") or "").strip()
        if not command or "\n" in command or "\r" in command:
            continue
        label = _state_readback_label(index, expectation, phase)
        lines.append(f'echo "{label}"')
        lines.append(command)
    return "\n".join(lines) + ("\n" if lines else "")


_RENODE_STIMULUS_PHASES = {"before_load", "before_run", "before_readback", "after_readback"}
_FORBIDDEN_RENODE_STIMULUS_SNIPPETS = (
    "\n",
    "\r",
    "LoadELF",
    "LoadPlatformDescription",
    "include",
    "mach create",
    "using ",
    "quit",
)
_ALLOWED_RENODE_STIMULUS_PATTERNS = (
    re.compile(r'^echo\s+"[A-Za-z0-9_ .:/+-]{1,160}"$'),
    re.compile(r'^emulation\s+RunFor\s+"(?:0\.\d+|[1-9]\d*(?:\.\d+)?)s"$'),
    re.compile(r'^(?:usart[1236]|uart[45])\s+WriteLine\s+"[A-Za-z0-9_ .:/+-]{1,160}"$'),
    re.compile(r'^(?:usart[1236]|uart[45])\s+WriteChar\s+(?:0x[0-9A-Fa-f]{1,2}|[0-9]{1,3})$'),
    re.compile(r'^gpioPort[A-I]\s+OnGPIO\s+(?:[0-9]|1[0-5])\s+(?:true|false|True|False)$'),
    re.compile(r'^nvic\s+OnGPIO\s+(?:[0-9]|[1-7][0-9]|8[01])\s+(?:true|false|True|False)$'),
    re.compile(r'^spi[123]\s+WriteByte\s+0x[0-9A-Fa-f]{1,2}\s+0x[0-9A-Fa-f]{1,2}$'),
)


def _raw_renode_stimulus_commands(case: Optional[dict[str, Any]]) -> list[Any]:
    if not isinstance(case, dict):
        return []
    commands = case.get("renode_stimulus_commands")
    if not isinstance(commands, list):
        renode_verification = case.get("renode_verification") or {}
        if isinstance(renode_verification, dict):
            commands = renode_verification.get("renode_stimulus_commands")
    return commands if isinstance(commands, list) else []


def _renode_stimulus_line(raw: Any) -> tuple[str, str] | None:
    if isinstance(raw, str):
        phase = "before_readback"
        command = raw
    elif isinstance(raw, dict):
        phase = str(raw.get("phase") or "before_readback").strip()
        command = str(raw.get("command") or raw.get("monitor_command") or "").strip()
    else:
        return None
    if phase not in _RENODE_STIMULUS_PHASES:
        return None
    if not command:
        return None
    if any(snippet in command for snippet in _FORBIDDEN_RENODE_STIMULUS_SNIPPETS):
        return None
    if re.match(r"^sysbus\s+Write(?:Byte|Word|DoubleWord|QuadWord)\b", command):
        return None
    if not any(pattern.match(command) for pattern in _ALLOWED_RENODE_STIMULUS_PATTERNS):
        return None
    return phase, command


def _renode_stimulus_commands(case: Optional[dict[str, Any]], phase: str) -> str:
    lines: list[str] = []
    for raw in _raw_renode_stimulus_commands(case):
        parsed = _renode_stimulus_line(raw)
        if parsed is None:
            continue
        command_phase, command = parsed
        if command_phase == phase:
            lines.append(command)
    return "\n".join(lines) + ("\n" if lines else "")


_RENODE_INTERNAL_EVENT_PHASES = {"before_readback"}
_RENODE_INTERNAL_EVENT_FORBIDDEN_FIELDS = {
    "address",
    "raw_monitor_command",
    "monitor_command",
    "command",
    "arbitrary_value",
    "ordered_log_text_to_emit",
    "semantic_pass_override",
}
_DMA_INTERNAL_EVENT_ALLOWLIST: dict[int, list[dict[str, Any]]] = {
    7: [
        {
            "event_kind": "bounded_dma_transfer_complete_event",
            "controller": "DMA1",
            "stream": "Stream5",
            "status_register": "HISR",
            "status_bit": 11,
            "irq": "DMA1_Stream5_IRQn",
            "irq_number": 16,
        }
    ],
    9: [
        {
            "event_kind": "bounded_dma_transfer_complete_event",
            "controller": "DMA1",
            "stream": "Stream6",
            "status_register": "HISR",
            "status_bit": 21,
            "irq": "DMA1_Stream6_IRQn",
            "irq_number": 17,
        }
    ],
    43: [
        {
            "event_kind": "bounded_dma_half_transfer_event",
            "controller": "DMA2",
            "stream": "Stream1",
            "status_register": "HISR",
            "status_bit": 4,
            "irq": "DMA2_Stream1_IRQn",
            "irq_number": 57,
        },
        {
            "event_kind": "bounded_dma_transfer_complete_event",
            "controller": "DMA2",
            "stream": "Stream1",
            "status_register": "HISR",
            "status_bit": 5,
            "irq": "DMA2_Stream1_IRQn",
            "irq_number": 57,
        },
    ],
}
_DMA_STATUS_REGISTER_ADDRESS = {
    ("DMA1", "HISR"): 0x40026004,
    ("DMA2", "HISR"): 0x40026404,
}


def _raw_renode_internal_events(case: Optional[dict[str, Any]]) -> list[Any]:
    if not isinstance(case, dict):
        return []
    events = case.get("renode_internal_events")
    if not isinstance(events, list):
        renode_verification = case.get("renode_verification") or {}
        if isinstance(renode_verification, dict):
            events = renode_verification.get("renode_internal_events")
    return events if isinstance(events, list) else []


def _case_id(case: Optional[dict[str, Any]]) -> int | None:
    if not isinstance(case, dict):
        return None
    raw_case_id = case.get("id", case.get("case_id"))
    try:
        return int(raw_case_id)
    except (TypeError, ValueError):
        return None


def _renode_internal_event_lines(case: Optional[dict[str, Any]], raw: Any) -> tuple[str, list[str]] | None:
    if not isinstance(case, dict) or not isinstance(raw, dict):
        return None
    if any(field in raw for field in _RENODE_INTERNAL_EVENT_FORBIDDEN_FIELDS):
        return None
    phase = str(raw.get("phase") or "before_readback").strip()
    if phase not in _RENODE_INTERNAL_EVENT_PHASES:
        return None
    case_id = _case_id(case)
    if case_id is None:
        return None
    allowed = _DMA_INTERNAL_EVENT_ALLOWLIST.get(case_id, [])
    event = {
        "event_kind": str(raw.get("event_kind") or "").strip(),
        "controller": str(raw.get("controller") or "").strip(),
        "stream": str(raw.get("stream") or "").strip(),
        "status_register": str(raw.get("status_register") or "").strip(),
        "status_bit": raw.get("status_bit"),
        "irq": str(raw.get("irq") or "").strip(),
        "irq_number": raw.get("irq_number"),
    }
    try:
        event["status_bit"] = int(event["status_bit"])
        event["irq_number"] = int(event["irq_number"])
    except (TypeError, ValueError):
        return None
    if event not in allowed:
        return None
    address = _DMA_STATUS_REGISTER_ADDRESS.get((event["controller"], event["status_register"]))
    if address is None:
        return None
    value = 1 << int(event["status_bit"])
    event_id = str(raw.get("event_id") or f"case{case_id}_{event['controller']}_{event['stream']}_{event['status_bit']}")
    event_id = re.sub(r"[^A-Za-z0-9_]+", "_", event_id).strip("_").upper()
    if not event_id:
        event_id = f"CASE{case_id}_DMA_EVENT"
    lines = [
        f'echo "CORE600_INTERNAL_EVENT_{event_id}"',
        f"sysbus WriteDoubleWord 0x{address:08X} 0x{value:X}",
        f"nvic OnGPIO {event['irq_number']} true",
        'emulation RunFor "0.005s"',
        f"nvic OnGPIO {event['irq_number']} false",
    ]
    return phase, lines


def _renode_internal_event_commands(case: Optional[dict[str, Any]], phase: str) -> str:
    lines: list[str] = []
    for raw in _raw_renode_internal_events(case):
        parsed = _renode_internal_event_lines(case, raw)
        if parsed is None:
            continue
        event_phase, event_lines = parsed
        if event_phase == phase:
            lines.extend(event_lines)
    return "\n".join(lines) + ("\n" if lines else "")


def _parse_state_readbacks(log_text: str) -> dict[str, int | None]:
    readbacks: dict[str, int | None] = {}
    current: str | None = None
    for line in log_text.splitlines():
        marker = re.search(r"\b(CORE600_READBACK_(?:BEFORE|AFTER)_[A-Z0-9_]+)\b", line)
        if marker:
            current = marker.group(1)
            readbacks[current] = None
            continue
        if current is None:
            continue
        value_match = re.search(r"\breturned\s+0x([0-9a-fA-F]+)\b", line, flags=re.IGNORECASE)
        if value_match:
            readbacks[current] = int(value_match.group(1), 16)
            current = None
            continue
        svd_value_match = re.search(
            r"\breturning\s+(?:a\s+)?value\s+from\s+SVD:\s*0x([0-9a-fA-F]+)\b",
            line,
            flags=re.IGNORECASE,
        )
        if svd_value_match:
            readbacks[current] = int(svd_value_match.group(1), 16)
            current = None
            continue
        plain_hex = re.fullmatch(r"\s*0x([0-9a-fA-F]+)\s*", line)
        if plain_hex:
            readbacks[current] = int(plain_hex.group(1), 16)
            current = None
            continue
        loose_hexes = re.findall(r"\b0x([0-9a-fA-F]+)\b", line)
        if loose_hexes:
            if "register" in line.lower() and len(loose_hexes) == 1:
                continue
            readbacks[current] = int(loose_hexes[-1], 16)
            current = None
            continue
        decimal = re.fullmatch(r"\s*(\d+)\s*", line)
        if decimal:
            readbacks[current] = int(decimal.group(1), 10)
            current = None
    return readbacks


def _parse_state_readback_mask(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            return None
    return None


def _format_state_readback_value(value: int | None) -> str | None:
    if value is None:
        return None
    return f"0x{value:x}"


def _evaluate_state_readbacks(
    case: Optional[dict[str, Any]],
    readbacks: dict[str, int | None],
) -> list[dict[str, Any]]:
    verdicts: list[dict[str, Any]] = []
    for index, expectation in enumerate(_state_readback_expectations(case), start=1):
        before_label = _state_readback_label(index, expectation, "BEFORE")
        after_label = _state_readback_label(index, expectation, "AFTER")
        before_value = readbacks.get(before_label)
        after_value = readbacks.get(after_label)
        relation = str(expectation.get("expected_relation") or "").strip()
        mask = _parse_state_readback_mask(expectation.get("expected_mask"))
        passed = False
        failure_reason = ""
        if mask is None:
            failure_reason = "invalid_expected_mask"
        elif after_value is None:
            failure_reason = "missing_after_readback"
        elif relation in {"nonzero", "bit_set"}:
            passed = (after_value & mask) != 0
        elif relation in {"zero", "bit_clear"}:
            passed = (after_value & mask) == 0
        elif relation == "changes_after_stimulus":
            if before_value is None:
                failure_reason = "missing_before_readback"
            else:
                passed = ((before_value ^ after_value) & mask) != 0
        else:
            failure_reason = "unsupported_expected_relation"

        if not passed and not failure_reason:
            failure_reason = "relation_not_satisfied"
        verdicts.append(
            {
                "target": str(expectation.get("target") or ""),
                "monitor_command": str(expectation.get("monitor_command") or ""),
                "expected_mask": expectation.get("expected_mask"),
                "expected_relation": relation,
                "before_label": before_label,
                "after_label": after_label,
                "before_value": _format_state_readback_value(before_value),
                "after_value": _format_state_readback_value(after_value),
                "passed": passed,
                "failure_reason": "" if passed else failure_reason,
            }
        )
    return verdicts


def _marker_state_bindings(case: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(case, dict):
        return []
    bindings = case.get("marker_state_bindings")
    if not isinstance(bindings, list):
        return []
    return [dict(item) for item in bindings if isinstance(item, dict)]


def _helper_check_pass_contract_enabled(case: Optional[dict[str, Any]]) -> bool:
    if not isinstance(case, dict):
        return False
    return (
        case.get("proof_contract_mode") == "helper_check_pass_v2"
        or case.get("proof_contract_mode") == "phase_checkpoint_v1"
        or case.get("validation_contract_mode") == "contract_patch_v2_helper_check_pass"
        or case.get("validation_contract_mode") == "contract_v2_phase_checkpoint_v1"
    )


def _expected_helper_check_pass_tokens(case: Optional[dict[str, Any]]) -> list[str]:
    tokens: list[str] = []
    for binding in _marker_state_bindings(case):
        token = str(binding.get("marker") or binding.get("token") or "").strip()
        if token:
            tokens.append(token)
    return tokens


def _has_ordered_token_list(found_tokens: list[str], expected_tokens: list[str]) -> bool:
    if not expected_tokens:
        return False
    cursor = 0
    for expected in expected_tokens:
        try:
            index = found_tokens.index(expected, cursor)
        except ValueError:
            return False
        cursor = index + 1
    return True


def _parse_marker_state_outputs(log_text: str) -> dict[str, dict[str, int]]:
    outputs: dict[str, dict[str, int]] = {}
    for line in log_text.splitlines():
        match = re.search(
            r"\bCORE600_MARKER_STATE\s+([A-Za-z0-9_:+./-]+)\s+"
            r"(0x[0-9A-Fa-f]+)\s+(0x[0-9A-Fa-f]+)\s+(0x[0-9A-Fa-f]+)\b",
            line,
        )
        if not match:
            continue
        token = match.group(1)
        outputs[token] = {
            "address": int(match.group(2), 16),
            "mask": int(match.group(3), 16),
            "value": int(match.group(4), 16),
        }
    return outputs


def _parse_helper_check_pass_outputs(log_text: str) -> list[str]:
    tokens: list[str] = []
    for line in log_text.splitlines():
        match = re.search(r"\bCORE600_CHECK_PASS\s+([A-Za-z0-9_:+./-]+)\b", line)
        if match:
            tokens.append(match.group(1))
    return tokens


def _binding_address(binding: dict[str, Any]) -> int | None:
    raw = binding.get("address", binding.get("monitor_address"))
    if raw is not None:
        try:
            return int(str(raw), 0)
        except ValueError:
            return None
    command = str(binding.get("monitor_command") or "")
    matches = re.findall(r"0x[0-9A-Fa-f]+", command)
    if not matches:
        return None
    return int(matches[-1], 16)


def _evaluate_marker_state_bindings(
    case: Optional[dict[str, Any]],
    marker_outputs: dict[str, dict[str, int]],
    readbacks: dict[str, int | None],
) -> list[dict[str, Any]]:
    verdicts: list[dict[str, Any]] = []
    for binding in _marker_state_bindings(case):
        token = str(binding.get("marker") or binding.get("token") or "").strip()
        target = str(binding.get("target") or "")
        relation = str(binding.get("expected_relation") or "").strip()
        mask = _parse_state_readback_mask(binding.get("expected_mask", binding.get("mask")))
        address = _binding_address(binding)
        output = marker_outputs.get(token)
        value = output.get("value") if output else None
        passed = False
        failure_reason = ""
        if not token:
            failure_reason = "missing_binding_marker"
        elif mask is None:
            failure_reason = "invalid_expected_mask"
        elif address is None:
            failure_reason = "invalid_marker_address"
        elif output is None:
            failure_reason = "missing_marker_state_output"
        elif output.get("address") != address:
            failure_reason = "marker_state_address_mismatch"
        elif output.get("mask") != mask:
            failure_reason = "marker_state_mask_mismatch"
        elif relation in {"nonzero", "bit_set"}:
            passed = (int(value) & mask) != 0
        elif relation in {"zero", "bit_clear"}:
            passed = (int(value) & mask) == 0
        elif relation == "changes_after_stimulus":
            expectation_index = binding.get("expectation_index")
            before_value = None
            try:
                index = int(expectation_index)
            except (TypeError, ValueError):
                index = 0
            if index > 0:
                expectations = _state_readback_expectations(case)
                if 1 <= index <= len(expectations):
                    before_label = _state_readback_label(index, expectations[index - 1], "BEFORE")
                    before_value = readbacks.get(before_label)
            if before_value is None:
                failure_reason = "missing_before_readback"
            else:
                passed = ((before_value ^ int(value)) & mask) != 0
        else:
            failure_reason = "unsupported_expected_relation"
        if not passed and not failure_reason:
            failure_reason = "relation_not_satisfied"
        verdicts.append(
            {
                "marker": token,
                "target": target,
                "address": f"0x{address:x}" if address is not None else None,
                "expected_mask": binding.get("expected_mask", binding.get("mask")),
                "expected_relation": relation,
                "observed_address": f"0x{output.get('address'):x}" if output else None,
                "observed_mask": f"0x{output.get('mask'):x}" if output else None,
                "observed_value": f"0x{value:x}" if value is not None else None,
                "passed": passed,
                "failure_reason": "" if passed else failure_reason,
            }
        )
    return verdicts


def _next200_readback_policy_diagnostics(
    case: Optional[dict[str, Any]],
    state_verdicts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expectations = _state_readback_expectations(case)
    diagnostics: list[dict[str, Any]] = []
    for expectation, verdict in zip(expectations, state_verdicts):
        annotation = expectation.get("next200_candidate_policy_annotation")
        if not isinstance(annotation, dict):
            continue
        candidate_id = str(annotation.get("candidate_id") or "")
        target = str(annotation.get("target") or verdict.get("target") or "")
        passed = bool(verdict.get("passed"))
        blockers: list[str] = []
        if candidate_id in {
            "next200_tim2_sr_blocked_timer_policy_v1",
            "next200_tim5_sr_event_policy_v1",
        }:
            blockers.append("timer_event_control_validation_missing")
        diagnostics.append(
            {
                "target": target,
                "candidate_id": candidate_id,
                "readback_passed": passed,
                "diagnostic_demote_eligible": not passed and not blockers,
                "must_still_require_complete_ordered_logs": True,
                "must_still_require_non_token_semantic_anchor": True,
                "blockers": blockers,
            }
        )
    return diagnostics


def _next200_timeout_postmortem_diagnostics(
    log_text: str,
    case: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    timeout_annotations = [
        annotation
        for annotation in _next200_policy_annotations(case)
        if annotation.get("candidate_policy_role") == "diagnostic_timeout_policy_pending_runner_support"
    ]
    if not timeout_annotations:
        return diagnostics
    primary_text = _extract_usart_terminal_output(log_text) or log_text
    ordered_logs = case.get("required_ordered_logs") if isinstance(case, dict) else []
    ordered_logs = ordered_logs if isinstance(ordered_logs, list) else []
    found_count = 0
    for token in ordered_logs:
        candidates = _ordered_log_candidates(str(token), case)
        if any(re.search(rf"\b{re.escape(candidate)}\b", primary_text, re.IGNORECASE) for candidate in candidates):
            found_count += 1
    complete = bool(ordered_logs) and found_count == len(ordered_logs)
    for annotation in timeout_annotations:
        diagnostics.append(
            {
                "candidate_id": str(annotation.get("candidate_id") or ""),
                "ordered_token_found_count": found_count,
                "required_ordered_log_count": len(ordered_logs),
                "complete_ordered_logs_before_timeout": complete,
                "timeout_auto_pass_allowed": False,
                "sim_ok_override_allowed": False,
            }
        )
    return diagnostics


def _readback_value_as_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            return None
    return None


def _has_target_bound_changed_state_readback_anchor(state_verdicts: list[dict[str, Any]]) -> bool:
    for verdict in state_verdicts:
        if not verdict.get("passed"):
            continue
        if not (verdict.get("target") and verdict.get("before_label") and verdict.get("after_label")):
            continue
        before_value = _readback_value_as_int(verdict.get("before_value"))
        after_value = _readback_value_as_int(verdict.get("after_value"))
        if before_value is None or after_value is None:
            continue
        if before_value != after_value:
            return True
    return False


# Map SVD register-level tags to required_control_pattern tokens
_SVD_TAG_TO_CONTROL_PATTERN: dict[str, list[str]] = {
    "CIRC": ["reentry_guard_on_ht_tc_back_to_back", "circular_mode_enabled"],
    "HTIE": ["half_transfer_interrupt", "ht_ready", "htie_enabled"],
    "TCIE": ["transfer_complete_interrupt", "tc_ready", "tcie_enabled"],
    "DMEIE": ["direct_mode_error_interrupt", "dmeie_enabled"],
    "TEIE": ["transfer_error_interrupt", "teie_enabled"],
    "DMAT": ["dma_transmitter_enabled", "usart_dma_tx"],
    "DMAR": ["dma_receiver_enabled", "usart_dma_rx"],
    "DBM": ["double_buffer_mode", "ownership_flip", "own_flip"],
    "SBK": ["send_break", "usart_break"],
    "PRFTEN": ["flash_prefetch_enabled"],
    "FS": ["fifo_status", "dma_fifo"],
    "M1A": ["memory1_address", "double_buffer"],
}


def _extract_register_semantic_anchors(log_text: str) -> tuple[list[str], list[str]]:
    """Extract register-level semantic signals from Renode WARNING lines.

    Returns (register_anchors, svd_tags):
    - register_anchors: e.g. ["WWDG:CR", "SYSCFG:EXTICR1", "ADC1:SQR2"]
    - svd_tags: e.g. ["CIRC", "HTIE", "DMEIE", "DMAT"]

    Two pattern sources:
    1. sysbus register lines: "register PERIPHERAL:REGISTER (0xADDR)"
    2. Peripheral offset lines: "dma2: Unhandled write ... Tags: TAG1, TAG2"
    """
    register_anchors: list[str] = []
    svd_tags: list[str] = []

    # Pattern 1: sysbus register access
    # e.g. [WARNING] sysbus: [cpu: 0x80001DC] WriteDoubleWord ... register WWDG:CR (0x40002C00)
    for match in re.finditer(
        r"register\s+(\w+:\w+)\s+\(0x[0-9a-fA-F]+\)", log_text
    ):
        anchor = match.group(1)
        if anchor not in register_anchors:
            register_anchors.append(anchor)

    # Pattern 2: Peripheral offset WARNING with SVD Tags
    # e.g. [WARNING] dma2: Unhandled write to offset 0x28 ... Tags: CIRC (0x1), HTIE (0x1).
    for match in re.finditer(
        r"Tags:\s*([^.\n]+)\.", log_text
    ):
        tags_str = match.group(1)
        for tag_match in re.finditer(r"([A-Z]\w+)\s*\(", tags_str):
            tag = tag_match.group(1)
            if tag != "RESERVED" and tag not in svd_tags:
                svd_tags.append(tag)

    return register_anchors, svd_tags


def _detect_forbidden_shortcuts(
    usart_output: str,
    case: Optional[dict[str, Any]],
) -> list[str]:
    """Detect forbidden shortcut patterns from failure_oracle in USART output.

    Returns list of matched forbidden pattern descriptions.
    """
    if not isinstance(case, dict):
        return []
    failure_oracle = case.get("failure_oracle") or []
    if not failure_oracle or not usart_output:
        return []

    forbidden: list[str] = []
    lower_output = usart_output.lower()

    # Map common failure_oracle concepts to detectable patterns in USART output
    _FAILURE_KEYWORD_MAP = {
        "duplicate consume": ["duplicate_consume", "double_consume", "re-consume"],
        "stale read": ["stale_read", "stale_half"],
        "without reentry": ["reentry_skipped", "no_reentry_guard"],
        "rollback": ["rollback", "index_rollback"],
        "drift": ["crc_drift", "checksum_drift"],
        "silent drop": ["silent_drop", "request_dropped"],
        "rewritten while": ["config_rewritten", "stream_reconfigured"],
        "before completion": ["resume_before_complete", "resume_before_flags"],
    }

    for failure_desc in failure_oracle:
        lower_desc = failure_desc.lower()
        for keyword, tokens in _FAILURE_KEYWORD_MAP.items():
            if keyword in lower_desc:
                for token in tokens:
                    if token in lower_output:
                        forbidden.append(failure_desc)
                        break

    return forbidden


def _semantic_oracle(
    log_text: str,
    case: Optional[dict[str, Any]],
) -> SemanticVerdict:
    """Perform semantic oracle analysis combining all signal sources.

    Integrates:
    1. USART terminal output (from showAnalyzer)
    2. Register-level semantic anchors (from SVD register names)
    3. SVD bitfield tags (from Renode WARNING lines)
    4. required_control_pattern matching
    5. failure_oracle forbidden shortcut detection
    """
    usart_output = _extract_usart_terminal_output(log_text)
    register_anchors, svd_tags = _extract_register_semantic_anchors(log_text)
    forbidden = _detect_forbidden_shortcuts(usart_output, case)
    readbacks = _parse_state_readbacks(log_text)
    state_verdicts = _evaluate_state_readbacks(case, readbacks)
    marker_outputs = _parse_marker_state_outputs(usart_output or log_text)
    helper_check_pass_tokens = _parse_helper_check_pass_outputs(usart_output or log_text)
    marker_state_verdicts = _evaluate_marker_state_bindings(case, marker_outputs, readbacks)
    next200_policy_diagnostics = _next200_readback_policy_diagnostics(case, state_verdicts)
    next200_policy_diagnostics.extend(_next200_timeout_postmortem_diagnostics(log_text, case))
    state_failures = [
        f"{item.get('target') or item.get('after_label')}: {item.get('failure_reason')}"
        for item in state_verdicts
        if not item.get("passed")
    ]
    marker_state_failures = [
        f"{item.get('marker') or item.get('target')}: {item.get('failure_reason')}"
        for item in marker_state_verdicts
        if not item.get("passed")
    ]

    # Match required_control_pattern from SVD tags and register anchors
    control_pattern_hits: list[str] = []
    if isinstance(case, dict):
        required_patterns = case.get("required_control_pattern") or []
        for pattern in required_patterns:
            lower_pattern = pattern.lower()
            # Check SVD tags
            for tag in svd_tags:
                mapped = _SVD_TAG_TO_CONTROL_PATTERN.get(tag, [])
                if any(lower_pattern == mp.lower() for mp in mapped):
                    if pattern not in control_pattern_hits:
                        control_pattern_hits.append(pattern)
                    break
            # Check register anchors
            for anchor in register_anchors:
                if any(part.lower() in lower_pattern for part in anchor.lower().split(":")):
                    if pattern not in control_pattern_hits:
                        control_pattern_hits.append(pattern)
                    break

    # Collect USART found tokens from ordered logs
    usart_tokens_found: list[str] = []
    if usart_output and isinstance(case, dict):
        required_logs = case.get("required_ordered_logs") or []
        for token in required_logs:
            candidates = _ordered_log_candidates(str(token), case)
            for candidate in candidates:
                if re.search(rf"\b{re.escape(candidate)}\b", usart_output, re.IGNORECASE):
                    usart_tokens_found.append(token)
                    break

    # Compute semantic score
    total_signals = 0
    positive_signals = 0

    if isinstance(case, dict):
        required_patterns = case.get("required_control_pattern") or []
        required_logs = case.get("required_ordered_logs") or []
        total_signals = len(required_patterns) + len(required_logs)
        positive_signals = len(control_pattern_hits) + len(usart_tokens_found)

    semantic_score = positive_signals / total_signals if total_signals > 0 else 0.0

    # For semantic_strict rows, ordered tokens alone are not enough: a token-only
    # stub can fake activity. Require the complete ordered log sequence and at
    # least one non-token semantic anchor from Renode/SVD/control-pattern evidence.
    if isinstance(case, dict) and _helper_check_pass_contract_enabled(case):
        anchor_present = _has_target_bound_changed_state_readback_anchor(state_verdicts)
        expected_check_pass_tokens = _expected_helper_check_pass_tokens(case)
        likely_correct = (
            len(forbidden) == 0
            and not state_failures
            and not marker_state_failures
            and _has_ordered_token_list(helper_check_pass_tokens, expected_check_pass_tokens)
            and anchor_present
        )
    elif isinstance(case, dict) and case.get("oracle_contract_mode") in _STRICT_ORACLE_CONTRACT_MODES:
        primary_text = usart_output or log_text
        anchor_present = bool(control_pattern_hits or register_anchors or svd_tags)
        if case.get("oracle_contract_mode") == "semantic_strict_state_readback_anchor_candidate_v1":
            anchor_present = anchor_present or _has_target_bound_changed_state_readback_anchor(state_verdicts)
        likely_correct = (
            len(forbidden) == 0
            and not state_failures
            and not marker_state_failures
            and _has_required_ordered_logs(primary_text, case)
            and anchor_present
        )
    else:
        # Determine likely_correct: no forbidden shortcuts and sufficient semantic signals
        likely_correct = (
            len(forbidden) == 0
            and not state_failures
            and not marker_state_failures
            and semantic_score >= 0.5
            and (bool(usart_tokens_found) or bool(control_pattern_hits))
        )

    return SemanticVerdict(
        usart_tokens_found=usart_tokens_found,
        register_anchors=register_anchors,
        svd_tags=svd_tags,
        forbidden_shortcuts=forbidden,
        control_pattern_hits=control_pattern_hits,
        state_readback_expectations=state_verdicts,
        state_readback_failures=state_failures,
        marker_state_bindings=marker_state_verdicts,
        marker_state_failures=marker_state_failures,
        helper_check_pass_tokens=helper_check_pass_tokens,
        semantic_score=round(semantic_score, 3),
        likely_correct=likely_correct,
        next200_policy_diagnostics=next200_policy_diagnostics,
    )


def _extract_mmio_addresses(log_text: str) -> list[int]:
    addresses: list[int] = []
    for match in re.finditer(r"\b0x4[0-9a-f]{7}\b", log_text, flags=re.IGNORECASE):
        try:
            addresses.append(int(match.group(0), 16))
        except ValueError:
            continue
    return addresses


def _is_setup_only_address(address: int) -> bool:
    return any(start <= address < end for start, end in SETUP_ONLY_ADDRESS_RANGES)


def _case_observable_hints(case: Optional[dict[str, Any]]) -> set[str]:
    if not isinstance(case, dict):
        return set()

    hints = set(FAMILY_OBSERVABLE_HINTS.get(str(case.get("family_id") or ""), set()))
    text_parts: list[str] = []
    for key in (
        "requirement",
        "family_id",
        "cluster_id",
    ):
        value = case.get(key)
        if isinstance(value, str):
            text_parts.append(value.lower())
    for key in ("required_control_pattern", "required_ordered_logs", "success_oracle", "failure_oracle"):
        value = case.get(key) or []
        if isinstance(value, list):
            text_parts.extend(str(item).lower() for item in value)

    combined = " ".join(text_parts)
    for token, mapped_hints in TOKEN_OBSERVABLE_HINTS.items():
        if token in combined:
            hints.update(mapped_hints)
    return hints


def _case_named_signal_patterns(case: Optional[dict[str, Any]]) -> list[str]:
    hints = _case_observable_hints(case)
    if not hints:
        return []
    patterns: list[str] = []
    for hint in sorted(hints):
        patterns.append(rf"\b{re.escape(hint)}\d*\b:.*\bWrite")
        patterns.append(rf"\b{re.escape(hint)}\d*\b:.*\bnot supported\b")
    return patterns


def _matches_any_pattern(log_text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, log_text, flags=re.IGNORECASE) for pattern in patterns)


def _ordered_log_candidates(token: str, case: Optional[dict[str, Any]]) -> tuple[str, ...]:
    family_id = str(case.get("family_id") or "") if isinstance(case, dict) else ""
    if family_id != "HF-A1":
        return (token,)
    return ORDERED_LOG_ALIASES.get(token, (token,))


def _has_required_ordered_logs(log_text: str, case: Optional[dict[str, Any]]) -> bool:
    if not isinstance(case, dict):
        return False
    required_logs = case.get("required_ordered_logs") or []
    if not isinstance(required_logs, list) or not required_logs:
        return False

    cursor = 0
    for token in required_logs:
        if not token:
            return False
        candidates = _ordered_log_candidates(str(token), case)
        alternation = "|".join(re.escape(candidate) for candidate in candidates)
        pattern = re.compile(rf"\b(?:{alternation})\b", flags=re.IGNORECASE)
        match = pattern.search(log_text, cursor)
        if not match:
            return False
        cursor = match.end()
    return True


def _has_observable_peripheral_activity(log_text: str, *, case: Optional[dict[str, Any]] = None) -> bool:
    """Best-effort Renode oracle for observable MMIO side effects.

    We intentionally accept both:
    - direct sysbus writes into the STM32 peripheral MMIO space (0x4000_0000...)
    - named peripheral warnings such as ``usart2: Unhandled write to offset ...``
    - USART terminal output (enabled via showAnalyzer in .resc)

    If USART terminal output is present, the code successfully wrote to a
    USART peripheral, which counts as observable peripheral activity even
    if no MMIO WARNING pattern was matched.
    """
    if not log_text:
        return False

    usart_output = _extract_usart_terminal_output(log_text)
    primary_text = usart_output or log_text
    secondary_text = log_text

    if isinstance(case, dict) and case.get("oracle_contract_mode") == "ordered_logs_required":
        return _has_required_ordered_logs(primary_text, case)

    # USART terminal output means firmware reached an observable USART write,
    # unless the case explicitly required stricter ordered-log semantics above.
    if usart_output:
        return True

    if not case:
        return _matches_any_pattern(log_text, OBSERVABLE_PERIPHERAL_PATTERNS)

    if _has_required_ordered_logs(primary_text, case):
        return True

    if _matches_any_pattern(secondary_text, _case_named_signal_patterns(case)):
        return True

    if _matches_any_pattern(secondary_text, SETUP_ONLY_NAMED_PATTERNS):
        non_setup_addrs = [addr for addr in _extract_mmio_addresses(secondary_text) if not _is_setup_only_address(addr)]
        if non_setup_addrs:
            return True
        if _matches_any_pattern(secondary_text, GENERIC_NON_SETUP_PERIPHERAL_PATTERNS):
            return True
        return False

    if _matches_any_pattern(secondary_text, OBSERVABLE_PERIPHERAL_PATTERNS):
        return True

    return False


def _load_result_rows(results_file: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not results_file.exists():
        return rows
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _dedupe_rows_by_case_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[int, dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []
    for row in rows:
        case_id = row.get("case_id")
        if isinstance(case_id, int):
            deduped[case_id] = row
        else:
            passthrough.append(row)
    return list(deduped.values()) + passthrough


def _sim_attempted(row: dict[str, Any]) -> bool:
    if row.get("infra_invalid"):
        return False
    if not row.get("compile_ok"):
        return False
    if row.get("sim_ok"):
        return True
    if str(row.get("sim_status") or "").strip():
        return True
    try:
        return float(row.get("sim_time_ms") or 0) > 0
    except (TypeError, ValueError):
        return False


def _scoreable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not row.get("infra_invalid")]


def _semantic_verdict_available(row: dict[str, Any]) -> bool:
    return isinstance(row.get("sim_verdict"), dict)


def _semantic_likely_correct(row: dict[str, Any]) -> bool:
    verdict = row.get("sim_verdict")
    return isinstance(verdict, dict) and bool(verdict.get("likely_correct"))


def _case_set_id_for_path(cases_path: Path) -> str:
    name = Path(cases_path).name
    if name.endswith(".cases.json"):
        return name[: -len(".cases.json")]
    return Path(cases_path).stem


def _default_output_dir(model: str, cases_path: Path) -> Path:
    model_name = model.rstrip("/").split("/")[-1].replace(" ", "_")
    case_set_id = _case_set_id_for_path(cases_path)
    return PROJECT_ROOT / "output_benchmark" / "core600" / model_name / case_set_id


def _candidate_protocol_path(cases_path: Path) -> Optional[Path]:
    name = cases_path.name
    if not name.endswith(".cases.json"):
        return None
    candidate = cases_path.with_name(f"{name[: -len('.cases.json')]}.protocol.json")
    return candidate if candidate.exists() else None


def _case_bundle_context_updates(cases_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    context_updates: dict[str, Any] = {}

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    if metadata:
        for key in (
            "status",
            "surface_id",
            "protocol_id",
            "protocol_type",
            "source_cases_path",
            "source_case_set_id",
            "oracle_contract_mode",
            "required_provenance_class",
            "clean_scored_rerun_allowed",
            "base_core600_eval_set_mutation_allowed",
            "promotion_allowed_count",
            "scored_truth_count",
            "eval_set_mutation_allowed_count",
        ):
            value = metadata.get(key)
            if value not in (None, "", [], {}):
                context_key = "case_set_status" if key == "status" else key
                context_updates[context_key] = value
        if "require_clean_scored_run" in metadata:
            context_updates["require_clean_scored_run"] = bool(metadata.get("require_clean_scored_run"))

    protocol_path = _candidate_protocol_path(cases_path)
    if protocol_path is not None:
        try:
            protocol_payload = json.loads(protocol_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            protocol_payload = {}
        if isinstance(protocol_payload, dict):
            context_updates["protocol_path"] = str(protocol_path)
            for key in (
                "protocol_id",
                "protocol_type",
                "resolved_model_id",
                "source_cases_path",
                "source_case_set_id",
                "required_provenance_class",
            ):
                value = protocol_payload.get(key)
                if value not in (None, "", [], {}):
                    context_updates[key] = value
            run_policy = protocol_payload.get("run_policy")
            if isinstance(run_policy, dict):
                if "require_clean_scored_run" in run_policy:
                    context_updates["require_clean_scored_run"] = bool(run_policy.get("require_clean_scored_run"))
                oracle_mode = run_policy.get("oracle_contract_mode")
                if oracle_mode not in (None, "", [], {}):
                    context_updates["oracle_contract_mode"] = oracle_mode

    source_cases_path = str(context_updates.get("source_cases_path") or "")
    if source_cases_path and "source_case_set_id" not in context_updates:
        context_updates["source_case_set_id"] = _case_set_id_for_path(Path(source_cases_path))
    return context_updates


_SECRET_KEY_FRAGMENTS = ("api_key", "apikey", "token", "secret")


def _load_run_context_bindings(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("run_context_bindings"), dict):
        payload = payload["run_context_bindings"]
    if not isinstance(payload, dict):
        raise ValueError(f"run context bindings must be a JSON object: {path}")
    bindings: dict[str, Any] = {}
    for key, value in payload.items():
        key_text = str(key)
        if any(fragment in key_text.lower() for fragment in _SECRET_KEY_FRAGMENTS):
            raise ValueError(f"run context binding refuses secret-like key: {key_text}")
        if isinstance(value, str) and re.search(r"\bsk-[A-Za-z0-9]{20,}\b", value):
            raise ValueError(f"run context binding refuses secret-like value for key: {key_text}")
        if value not in (None, "", [], {}):
            bindings[key_text] = value
    return bindings


def ensure_output_case_set(
    output_dir: Path,
    *,
    model: str,
    case_set_id: str,
    cases_path: Path,
    context_updates: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Persist and validate the case set bound to an output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    context_path = output_dir / RUN_CONTEXT_FILENAME
    expected = {
        "model": model,
        "case_set_id": case_set_id,
        "cases_path": str(cases_path),
    }
    if context_updates:
        for key, value in context_updates.items():
            if value not in (None, "", [], {}):
                expected[key] = value
    if context_path.exists():
        existing = json.loads(context_path.read_text(encoding="utf-8"))
        if existing.get("case_set_id") != expected.get("case_set_id") or existing.get("cases_path") != str(cases_path):
            raise ValueError(
                f"output directory {output_dir} is already bound to "
                f"{existing.get('case_set_id')} ({existing.get('cases_path')})"
            )
        merged = dict(existing)
        merged.update(expected)
        if merged != existing:
            context_path.write_text(
                json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        return merged
    context_path.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return expected


def load_checkpoint(output_dir: Path) -> set:
    """Load already completed case IDs from results.jsonl."""
    results_file = output_dir / "results.jsonl"
    done = {
        row["case_id"]
        for row in _dedupe_rows_by_case_id(_load_result_rows(results_file))
        if isinstance(row.get("case_id"), int)
    }
    return done


def save_result(output_dir: Path, result: CaseResult):
    """Append a single result to results.jsonl."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.jsonl"
    result.timestamp = datetime.now().isoformat()
    with open(results_file, "a") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def generate_summary(output_dir: Path, model: str, total_cases: int):
    """Generate summary JSON from results.jsonl."""
    results_file = output_dir / "results.jsonl"
    if not results_file.exists():
        return

    results = _dedupe_rows_by_case_id(_load_result_rows(results_file))

    total = len(results)
    scoreable = _scoreable_rows(results)
    scoreable_total = len(scoreable)
    infra_invalid = total - scoreable_total
    gen_ok = sum(1 for r in scoreable if r.get("gen_ok"))
    compile_ok = sum(1 for r in scoreable if r.get("compile_ok"))
    sim_ok = sum(1 for r in scoreable if r.get("sim_ok"))
    sim_attempted = sum(1 for r in scoreable if _sim_attempted(r))

    gen_rate = gen_ok / scoreable_total * 100 if scoreable_total else 0
    compile_rate = compile_ok / scoreable_total * 100 if scoreable_total else 0
    sim_rate = sim_ok / sim_attempted * 100 if sim_attempted else 0
    pass_rate = sim_ok / scoreable_total * 100 if scoreable_total else 0
    semantic_verdicts = sum(1 for r in scoreable if _semantic_verdict_available(r))
    semantic_pass = sum(1 for r in scoreable if _semantic_likely_correct(r))

    summary = {
        "model": model,
        "total_cases": total_cases,
        "completed": total,
        "metrics": {
            "scoreable_cases": scoreable_total,
            "infra_invalid": infra_invalid,
            "gen_ok": gen_ok,
            "compile_ok": compile_ok,
            "sim_ok": sim_ok,
            "sim_attempted": sim_attempted,
            "gen_rate_pct": round(gen_rate, 2),
            "compile_rate_pct": round(compile_rate, 2),
            "sim_rate_pct": round(sim_rate, 2),
            "pass_rate_pct": round(pass_rate, 2),
            "activity_sim_pass_rate_pct": round(pass_rate, 2),
            "semantic_verdicts_available": semantic_verdicts,
            "semantic_likely_correct": semantic_pass,
            "semantic_coverage_pct": round(semantic_verdicts / scoreable_total * 100, 2)
            if scoreable_total
            else 0.0,
            "semantic_pass_rate_pct": round(semantic_pass / scoreable_total * 100, 2)
            if scoreable_total
            else 0.0,
        },
        "by_level": {},
        "by_family": {},
        "timestamp": datetime.now().isoformat(),
    }

    # Per-level
    for level in ["L4", "L5"]:
        subset = [r for r in scoreable if r.get("level") == level]
        if subset:
            s_compile = sum(1 for r in subset if r.get("compile_ok"))
            s_sim = sum(1 for r in subset if r.get("sim_ok"))
            s_semantic_verdicts = sum(1 for r in subset if _semantic_verdict_available(r))
            s_semantic_pass = sum(1 for r in subset if _semantic_likely_correct(r))
            summary["by_level"][level] = {
                "total": len(subset),
                "compile_ok": s_compile,
                "compile_rate_pct": round(s_compile / len(subset) * 100, 2),
                "sim_ok": s_sim,
                "pass_rate_pct": round(s_sim / len(subset) * 100, 2),
                "activity_sim_pass_rate_pct": round(s_sim / len(subset) * 100, 2),
                "semantic_verdicts_available": s_semantic_verdicts,
                "semantic_likely_correct": s_semantic_pass,
                "semantic_coverage_pct": round(s_semantic_verdicts / len(subset) * 100, 2),
                "semantic_pass_rate_pct": round(s_semantic_pass / len(subset) * 100, 2),
            }

    # Per-family
    for fam in FAMILY_ORDER:
        subset = [r for r in scoreable if r.get("family_id") == fam]
        if subset:
            s_compile = sum(1 for r in subset if r.get("compile_ok"))
            s_sim = sum(1 for r in subset if r.get("sim_ok"))
            s_semantic_verdicts = sum(1 for r in subset if _semantic_verdict_available(r))
            s_semantic_pass = sum(1 for r in subset if _semantic_likely_correct(r))
            summary["by_family"][fam] = {
                "total": len(subset),
                "compile_ok": s_compile,
                "compile_rate_pct": round(s_compile / len(subset) * 100, 2),
                "sim_ok": s_sim,
                "pass_rate_pct": round(s_sim / len(subset) * 100, 2),
                "activity_sim_pass_rate_pct": round(s_sim / len(subset) * 100, 2),
                "semantic_verdicts_available": s_semantic_verdicts,
                "semantic_likely_correct": s_semantic_pass,
                "semantic_coverage_pct": round(s_semantic_verdicts / len(subset) * 100, 2),
                "semantic_pass_rate_pct": round(s_semantic_pass / len(subset) * 100, 2),
            }

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def run_eval(
    model: str,
    cases: list,
    api_key: str,
    output_dir: Path,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    skip_sim: bool = False,
    semantic_oracle: bool = False,
    api_url: Optional[str] = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    llm_timeout: int = LLM_TIMEOUT,
    llm_max_retries: int = LLM_MAX_RETRIES,
    llm_failure_stop_threshold: int = 0,
    compile_timeout: int = COMPILE_TIMEOUT,
    renode_timeout_floor: int = RENODE_TIMEOUT_FLOOR,
    trace_peripheral_access: bool = RENODE_TRACE_PERIPHERAL_ACCESS,
):
    """Run evaluation for a single model."""
    preflight_reasons = run_benchmark_preflight(skip_sim=skip_sim)
    if preflight_reasons:
        raise BenchmarkPreflightError(preflight_reasons)

    done = load_checkpoint(output_dir)
    print(f"[core600] model={model}, output={output_dir}")
    print(f"[core600] total cases={len(cases)}, already done={len(done)}")

    end = end_idx or len(cases)
    todo = [c for c in cases[start_idx:end] if c["id"] not in done]
    print(f"[core600] remaining={len(todo)}, range=[{start_idx}, {end})")

    build_dir = output_dir / "build"
    compile_log_dir = output_dir / "compile_logs"
    elf_output_dir = output_dir / "elf"
    renode_log_dir = output_dir / "renode_logs"

    save_lock = Lock()
    completed_count = [0]
    consecutive_llm_failures = [0]

    def _process_case(i: int, case: dict) -> CaseResult:
        cid = case["id"]
        req = case["requirement"]
        prompt_req = _requirement_with_generation_prompt_patch(case)
        level = case.get("level", "?")
        fam = case.get("family_id", "?")
        cluster = case.get("cluster_id", "?")
        case_audit_fields = {
            "oracle_contract_mode": str(case.get("oracle_contract_mode") or ""),
            "source_draft_case_id": str(case.get("source_draft_case_id") or ""),
            "target_eval_surface_status": str(case.get("target_eval_surface_status") or ""),
            "promotion_candidate": case.get("promotion_candidate")
            if isinstance(case.get("promotion_candidate"), bool)
            else None,
            "scored_truth": case.get("scored_truth")
            if isinstance(case.get("scored_truth"), bool)
            else None,
            "eval_set_mutation_allowed": case.get("eval_set_mutation_allowed")
            if isinstance(case.get("eval_set_mutation_allowed"), bool)
            else None,
        }

        print(f"[{i+1}/{len(todo)}] case={cid} level={level} fam={fam}", end=" ", flush=True)

        try:
            # 1. Generate
            deterministic_template_code = case.get("deterministic_template_code")
            generation_source = "llm"
            model_generation_started = True
            if isinstance(deterministic_template_code, str) and deterministic_template_code.strip():
                code = deterministic_template_code.strip()
                gen_ms = 0.0
                generation_source = "deterministic_template"
                model_generation_started = False
                valid, gen_errors = _validate_code(
                    code,
                    required_ordered_logs=case.get("required_ordered_logs"),
                    source_static_gate=case.get("source_static_gate"),
                    proof_contract_mode=case.get("proof_contract_mode"),
                )
                gen_ok = valid
            else:
                gen_ok, code, gen_ms, gen_errors = call_siliconflow(
                    model, prompt_req, api_key,
                    max_retries=llm_max_retries,
                    timeout=llm_timeout,
                    required_ordered_logs=case.get("required_ordered_logs"),
                    marker_state_bindings=case.get("marker_state_bindings"),
                    proof_contract_mode=case.get("proof_contract_mode"),
                    source_static_gate=case.get("source_static_gate"),
                    api_url=api_url,
                )
            if not gen_ok:
                if isinstance(code, str) and code.strip():
                    build_dir.mkdir(parents=True, exist_ok=True)
                    (build_dir / f"core600_{cid}.generation_failed.c").write_text(code, encoding="utf-8")
                result = CaseResult(
                    case_id=cid, level=level, family_id=fam, cluster_id=cluster,
                    requirement=req, gen_ok=False, gen_time_ms=gen_ms,
                    error_category="template" if generation_source == "deterministic_template" else "llm",
                    error_reason=_compact_llm_error_reason(gen_errors),
                    generation_source=generation_source,
                    model_generation_started=model_generation_started,
                    **case_audit_fields,
                )
                print(f"GEN_FAIL ({';'.join(gen_errors[:2])})")
                return result

            # 2. Compile (with hw_context prepended)
            comp_ok, comp_ms, warnings, errors, err_msgs, elf = compile_code(
                code, req, cid, build_dir, compile_log_dir, elf_output_dir,
                usart_force=bool(case.get("required_ordered_logs")),
                marker_state_bindings=case.get("marker_state_bindings"),
                proof_contract_mode=case.get("proof_contract_mode"),
                compile_timeout=compile_timeout,
            )
            if not comp_ok:
                infra_invalid = any(msg in INFRA_INVALID_REASONS for msg in err_msgs)
                result = CaseResult(
                    case_id=cid, level=level, family_id=fam, cluster_id=cluster,
                    requirement=req, gen_ok=True, gen_time_ms=gen_ms,
                    compile_ok=False, compile_time_ms=comp_ms,
                    compile_errors=errors, compile_warnings=warnings,
                    error_category="infra" if infra_invalid else "compile",
                    error_reason=_compact_compile_error_reason(err_msgs),
                    infra_invalid=infra_invalid,
                    generation_source=generation_source,
                    model_generation_started=model_generation_started,
                    **case_audit_fields,
                )
                print(f"COMPILE_FAIL (errors={errors})")
                return result

            # 3. Simulate (optional)
            if skip_sim or not elf:
                result = CaseResult(
                    case_id=cid, level=level, family_id=fam, cluster_id=cluster,
                    requirement=req, gen_ok=True, gen_time_ms=gen_ms,
                    compile_ok=True, compile_time_ms=comp_ms,
                    compile_warnings=warnings,
                    generation_source=generation_source,
                    model_generation_started=model_generation_started,
                    **case_audit_fields,
                )
                print(f"COMPILE_OK (sim skipped)")
                return result

            sim_ok, sim_ms, sim_status, err_cat, err_reason, sim_verdict = simulate_elf(
                elf, cid, renode_log_dir, SIM_SECONDS, case=case,
                semantic_oracle=semantic_oracle,
                renode_timeout_floor=renode_timeout_floor,
                trace_peripheral_access=trace_peripheral_access,
            )
            result = CaseResult(
                case_id=cid, level=level, family_id=fam, cluster_id=cluster,
                requirement=req, gen_ok=True, gen_time_ms=gen_ms,
                compile_ok=True, compile_time_ms=comp_ms,
                compile_warnings=warnings,
                sim_ok=sim_ok, sim_time_ms=sim_ms, sim_status=sim_status,
                error_category=err_cat, error_reason=err_reason,
                infra_invalid=err_reason in INFRA_INVALID_REASONS or err_cat == "infra",
                sim_verdict=sim_verdict,
                generation_source=generation_source,
                model_generation_started=model_generation_started,
                **case_audit_fields,
            )
            status = "PASS" if sim_ok else f"SIM_FAIL({sim_status})"
            print(f"{status}")
            return result

        except Exception as e:
            print(f"UNHANDLED_EXCEPTION: {e}")
            traceback.print_exc()
            return CaseResult(
                case_id=cid, level=level, family_id=fam, cluster_id=cluster,
                requirement=req, gen_ok=False, gen_time_ms=0,
                error_category="unhandled", error_reason=str(e)[:200],
                **case_audit_fields,
            )

    if concurrency <= 1:
        # Serial mode (original behavior)
        for i, case in enumerate(todo):
            result = _process_case(i, case)
            with save_lock:
                save_result(output_dir, result)
                completed_count[0] += 1
                if result.error_category == "llm" and result.gen_ok is False:
                    consecutive_llm_failures[0] += 1
                else:
                    consecutive_llm_failures[0] = 0
                if completed_count[0] % 50 == 0:
                    generate_summary(output_dir, model, len(cases))
                if (
                    llm_failure_stop_threshold > 0
                    and consecutive_llm_failures[0] >= llm_failure_stop_threshold
                ):
                    print(
                        "[core600] STOP: consecutive LLM failures reached "
                        f"{consecutive_llm_failures[0]} "
                        f"(threshold={llm_failure_stop_threshold})"
                    )
                    break
    else:
        # Concurrent mode
        print(f"[core600] concurrency={concurrency}")
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_process_case, i, case): (i, case)
                       for i, case in enumerate(todo)}
            for future in as_completed(futures):
                i, case = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    cid = case["id"]
                    result = CaseResult(
                        case_id=cid, level=case.get("level", "?"),
                        family_id=case.get("family_id", "?"),
                        cluster_id=case.get("cluster_id", "?"),
                        requirement=case["requirement"],
                        gen_ok=False, gen_time_ms=0,
                        error_category="unhandled", error_reason=str(e)[:200],
                        oracle_contract_mode=str(case.get("oracle_contract_mode") or ""),
                        source_draft_case_id=str(case.get("source_draft_case_id") or ""),
                        target_eval_surface_status=str(case.get("target_eval_surface_status") or ""),
                        promotion_candidate=case.get("promotion_candidate")
                        if isinstance(case.get("promotion_candidate"), bool)
                        else None,
                        scored_truth=case.get("scored_truth")
                        if isinstance(case.get("scored_truth"), bool)
                        else None,
                        eval_set_mutation_allowed=case.get("eval_set_mutation_allowed")
                        if isinstance(case.get("eval_set_mutation_allowed"), bool)
                        else None,
                    )
                with save_lock:
                    save_result(output_dir, result)
                    completed_count[0] += 1
                    if completed_count[0] % 50 == 0:
                        generate_summary(output_dir, model, len(cases))

    # Final summary
    summary = generate_summary(output_dir, model, len(cases))
    if summary:
        print(f"\n[core600] FINAL: {json.dumps(summary['metrics'], indent=2)}")


def main():
    parser = argparse.ArgumentParser(description="Core600 Large-Scale Benchmark Runner")
    parser.add_argument("--model", required=True, help="LLM model ID")
    parser.add_argument("--cases", default="data/benchmarks/core600_eval_set.json",
                        help="Path to eval set JSON")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: output_benchmark/core600/<model_name>)")
    parser.add_argument("--start", type=int, default=0, help="Start case index")
    parser.add_argument("--end", type=int, default=None, help="End case index (exclusive)")
    parser.add_argument("--skip-sim", action="store_true", help="Skip Renode simulation")
    parser.add_argument("--semantic-oracle", action="store_true",
                        help="Enable semantic oracle: sim_ok requires likely_correct verdict. "
                        "Rows with oracle_contract_mode=semantic_strict are always gated.")
    parser.add_argument("--api-key", default=None, help="API key (or set SF_API_KEY env)")
    parser.add_argument("--api-url", default=None,
                        help="API base URL (default: SiliconFlow direct). "
                        "For LiteLLM proxies use the full chat completions endpoint URL.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"Number of concurrent API calls (default: {DEFAULT_CONCURRENCY})")
    parser.add_argument("--llm-timeout", type=int, default=LLM_TIMEOUT,
                        help=f"Per-request LLM timeout in seconds (default: {LLM_TIMEOUT})")
    parser.add_argument("--llm-max-retries", type=int, default=LLM_MAX_RETRIES,
                        help=f"Maximum LLM retries per case (default: {LLM_MAX_RETRIES})")
    parser.add_argument("--llm-failure-stop-threshold", type=int, default=0,
                        help=(
                            "Stop serial execution after this many consecutive LLM generation "
                            "failures. 0 disables the guard (default: 0)."
                        ))
    parser.add_argument("--compile-timeout", type=int, default=COMPILE_TIMEOUT,
                        help=f"Per-case ARM GCC compile timeout in seconds (default: {COMPILE_TIMEOUT})")
    parser.add_argument("--renode-timeout-floor", type=int, default=RENODE_TIMEOUT_FLOOR,
                        help=(
                            "Minimum Renode subprocess timeout in seconds "
                            f"(default: {RENODE_TIMEOUT_FLOOR})"
                        ))
    parser.add_argument("--renode-trace-peripheral-access", action="store_true", default=RENODE_TRACE_PERIPHERAL_ACCESS,
                        help=(
                            "Enable sysbus LogAllPeripheralsAccess for diagnostic reruns. "
                            "This improves MMIO evidence but can slow Renode and enlarge logs."
                        ))
    parser.add_argument("--run-context-bindings", default=None,
                        help=(
                            "Optional JSON file containing non-secret run_context_bindings to merge "
                            "into run_context.json before execution."
                        ))

    args = parser.parse_args()

    # API key resolution
    api_key = args.api_key or SF_API_KEY or os.getenv("SILICONFLOW_API_KEY", "")
    if not api_key:
        print("ERROR: No API key provided. Set SF_API_KEY env or use --api-key.")
        sys.exit(1)

    # Load cases
    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"ERROR: Cases file not found: {cases_path}")
        sys.exit(1)
    with open(cases_path) as f:
        data = json.load(f)
    cases = data.get("cases", [])
    print(f"[core600] Loaded {len(cases)} cases from {cases_path}")

    # Output dir
    case_set_id = _case_set_id_for_path(cases_path)
    output_dir = Path(args.output) if args.output else _default_output_dir(args.model, cases_path)
    run_context_bindings = _load_run_context_bindings(
        Path(args.run_context_bindings) if args.run_context_bindings else None
    )
    ensure_output_case_set(
        output_dir,
        model=args.model,
        case_set_id=case_set_id,
        cases_path=cases_path,
        context_updates={
            **_case_bundle_context_updates(cases_path, data),
            **run_context_bindings,
            "api_url": args.api_url,
            "concurrency": args.concurrency,
            "llm_timeout": args.llm_timeout,
            "llm_max_retries": args.llm_max_retries,
            "llm_failure_stop_threshold": args.llm_failure_stop_threshold,
            "compile_timeout": args.compile_timeout,
            "renode_timeout_floor": args.renode_timeout_floor,
            "renode_trace_peripheral_access": args.renode_trace_peripheral_access,
        },
    )

    try:
        run_eval(
            model=args.model,
            cases=cases,
            api_key=api_key,
            output_dir=output_dir,
            start_idx=args.start,
            end_idx=args.end,
            skip_sim=args.skip_sim,
            semantic_oracle=args.semantic_oracle,
            api_url=args.api_url,
            concurrency=args.concurrency,
            llm_timeout=args.llm_timeout,
            llm_max_retries=args.llm_max_retries,
            llm_failure_stop_threshold=args.llm_failure_stop_threshold,
            compile_timeout=args.compile_timeout,
            renode_timeout_floor=args.renode_timeout_floor,
            trace_peripheral_access=args.renode_trace_peripheral_access,
        )
    except BenchmarkPreflightError as exc:
        print(f"ERROR: {exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
