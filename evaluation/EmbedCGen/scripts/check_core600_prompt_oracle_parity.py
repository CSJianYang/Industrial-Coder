#!/usr/bin/env python3
"""Preflight scored prompt/oracle parity for Core600 case bundles."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _flatten_prompt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_flatten_prompt(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return "\n".join(_flatten_prompt(v) for v in value)
    return str(value)


def model_facing_prompt_text(case: dict[str, Any]) -> str:
    ordered_logs = ""
    if case.get("proof_contract_mode") != "phase_checkpoint_v1":
        ordered_logs = "\n".join(str(token) for token in case.get("required_ordered_logs") or [])
    return "\n".join(
        part
        for part in (
            _flatten_prompt(case.get("requirement")),
            _flatten_prompt(case.get("generation_prompt_patch")),
            ordered_logs,
        )
        if part
    )


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip(), 0)
    except ValueError:
        return None


def _hex_variants(value: str) -> set[str]:
    parsed = _parse_int(value)
    variants = {str(value).lower()}
    if parsed is not None:
        variants.update({hex(parsed), f"0x{parsed:04x}", f"0x{parsed:08x}", str(parsed)})
    return {item.lower() for item in variants if item}


def _contains_tokenish(text: str, needle: str) -> bool:
    needle = str(needle or "").strip().lower()
    if not needle:
        return True
    if needle in text:
        return True
    return False


def _contains_mask(text: str, mask: Any) -> bool:
    for variant in _hex_variants(str(mask or "")):
        if re.search(rf"(?<![0-9a-fx]){re.escape(variant)}(?![0-9a-f])", text):
            return True
    return False


def _command_address(command: str) -> str | None:
    matches = re.findall(r"0x[0-9a-fA-F]+", str(command or ""))
    return matches[-1].lower() if matches else None


def _relation_visible(text: str, relation: str) -> bool:
    relation = str(relation or "").strip().lower()
    if not relation:
        return True
    if relation in text:
        return True
    aliases = {
        "bit_set": ["bit set", "must be set", "must be nonzero", "nonzero"],
        "nonzero": ["nonzero", "non-zero", "must be nonzero"],
        "bit_clear": ["bit clear", "must be clear", "must be zero"],
        "zero": ["zero", "must be zero"],
        "changes_after_stimulus": [
            "changes after stimulus",
            "change after stimulus",
            "must differ from before",
            "differs from before",
        ],
    }
    return any(alias in text for alias in aliases.get(relation, []))


def _final_state_visible(text: str, relation: str) -> bool:
    relation = str(relation or "").strip().lower()
    if "expected final state" in text:
        return True
    if relation in {"bit_set", "nonzero"}:
        return "final masked readback" in text and ("nonzero" in text or "bit set" in text)
    if relation in {"bit_clear", "zero"}:
        return "final masked readback" in text and ("zero" in text or "bit clear" in text)
    if relation == "changes_after_stimulus":
        return "final masked readback" in text and (
            "before-elf" in text or "before readback" in text or "before-run" in text
        )
    return bool(relation)


def _stable_scored(case: dict[str, Any]) -> bool:
    if case.get("diagnostic_not_scored") is True:
        return False
    if case.get("counted_as_model_score") is False:
        return False
    return True


def _case_id(case: dict[str, Any]) -> Any:
    return case.get("id", case.get("case_id", "<unknown>"))


def check_case(case: dict[str, Any]) -> list[dict[str, Any]]:
    if case.get("proof_contract_mode") == "phase_checkpoint_v1":
        return check_case_visibility_proof(case)
    prompt_text = model_facing_prompt_text(case).lower()
    issues: list[dict[str, Any]] = []
    for index, expectation in enumerate(case.get("state_readback_expectations") or [], start=1):
        target = str(expectation.get("target") or "")
        command = str(expectation.get("monitor_command") or "")
        address = _command_address(command)
        mask = expectation.get("expected_mask")
        relation = str(expectation.get("expected_relation") or "")

        missing: list[str] = []
        if not _contains_tokenish(prompt_text, target):
            missing.append("target")
        if command and command.lower() not in prompt_text and not (address and address in prompt_text):
            missing.append("monitor_command_or_address")
        if mask is None or not _contains_mask(prompt_text, mask):
            missing.append("mask")
        if not _relation_visible(prompt_text, relation):
            missing.append("expected_relation")
        if not _final_state_visible(prompt_text, relation):
            missing.append("expected_final_state")

        if missing:
            issues.append(
                {
                    "case_id": _case_id(case),
                    "expectation_index": index,
                    "target": target,
                    "missing": missing,
                }
            )
    return issues


def check_case_visibility_proof(case: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_text = model_facing_prompt_text(case).lower()
    prompt_phrases = case.get("prompt_phrases")
    prompt_phrases = prompt_phrases if isinstance(prompt_phrases, dict) else {}
    proofs = {
        str(item.get("target_key")): item
        for item in case.get("visibility_proofs") or []
        if isinstance(item, dict) and item.get("target_key")
    }
    issues: list[dict[str, Any]] = []
    for index, expectation in enumerate(case.get("state_readback_expectations") or [], start=1):
        key = str(expectation.get("registry_target_key") or "")
        target = str(expectation.get("target") or key)
        proof = proofs.get(key)
        missing: list[str] = []
        if not proof:
            missing.append("visibility_proof")
        else:
            level = str(proof.get("visibility_level") or "")
            phrase_id = str(proof.get("prompt_phrase_id") or "")
            phrase = str(prompt_phrases.get(phrase_id) or "")
            semantic_name = str(proof.get("register_bit_semantic_name") or "")
            if level not in {"semantic_visible", "numeric_mask_visible", "semantic_context_derivable"}:
                missing.append("stable_semantic_visibility")
            if proof.get("stable_score_allowed") is False:
                missing.append("stable_score_allowed")
            if not phrase_id or not phrase:
                missing.append("prompt_phrase")
            elif phrase.lower() not in prompt_text:
                missing.append("prompt_phrase_text")
            if not semantic_name:
                missing.append("register_bit_semantic_name")
            elif semantic_name.lower() not in prompt_text and phrase.lower() not in prompt_text:
                missing.append("semantic_name_or_phrase")
            if level == "numeric_mask_visible":
                command = str(expectation.get("monitor_command") or "")
                address = _command_address(command)
                mask = expectation.get("expected_mask")
                if command and command.lower() not in prompt_text and not (address and address in prompt_text):
                    missing.append("monitor_command_or_address")
                if mask is None or not _contains_mask(prompt_text, mask):
                    missing.append("mask")
        if missing:
            issues.append(
                {
                    "case_id": _case_id(case),
                    "expectation_index": index,
                    "target": target,
                    "registry_target_key": key,
                    "missing": missing,
                }
            )
    return issues


def check_bundle(payload: dict[str, Any]) -> dict[str, Any]:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("case bundle must contain a cases list")

    stable_cases = [case for case in cases if isinstance(case, dict) and _stable_scored(case)]
    diagnostic_cases = [case for case in cases if isinstance(case, dict) and not _stable_scored(case)]

    hidden_issues: list[dict[str, Any]] = []
    stable_partition_issues: list[dict[str, Any]] = []
    diagnostic_partition_issues: list[dict[str, Any]] = []

    for case in stable_cases:
        hidden_issues.extend(check_case(case))
        if case.get("diagnostic_not_scored") is True or case.get("counted_as_model_score") is False:
            stable_partition_issues.append({"case_id": _case_id(case), "issue": "stable_not_scoreable"})

    for case in diagnostic_cases:
        if case.get("diagnostic_not_scored") is not True and case.get("counted_as_model_score") is not False:
            diagnostic_partition_issues.append({"case_id": _case_id(case), "issue": "ambiguous_diagnostic_partition"})

    hidden_count = len(hidden_issues)
    blockers: list[str] = []
    if hidden_count:
        blockers.append("hidden_scored_readback_target")
    if stable_partition_issues or diagnostic_partition_issues:
        blockers.append("stable_diagnostic_partition_mismatch")

    return {
        "case_set_id": payload.get("case_set_id"),
        "case_count": len(cases),
        "stable_case_count": len(stable_cases),
        "diagnostic_case_count": len(diagnostic_cases),
        "hidden_scored_target_count": hidden_count,
        "hidden_scored_targets": hidden_issues[:50],
        "stable_partition_issues": stable_partition_issues[:50],
        "diagnostic_partition_issues": diagnostic_partition_issues[:50],
        "prompt_oracle_parity_pass": hidden_count == 0,
        "stable_diagnostic_partition_pass": not stable_partition_issues and not diagnostic_partition_issues,
        "blockers": blockers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = check_bundle(_load_json(args.cases))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            f.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["blockers"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
