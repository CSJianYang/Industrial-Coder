from scripts.core600_bench_runner import (
    _missing_required_trace_markers,
    _source_gate_repair_budget,
    _source_static_gate_errors,
)


def test_between_marker_gate_accepts_interleaved_write() -> None:
    source = """
    #include <stdint.h>
    void trace(const char *);
    int main(void) {
        USART2->CR1 |= (1u << 3);
        trace("A");
        GPIOA->ODR |= 0x20u;
        trace("B");
        trace("DONE");
        return 0;
    }
    """
    gate = {
        "mode": "mainline_ordered_logs_v1",
        "source_gate_repair_budget": 2,
        "target_bound_state_anchor_required": True,
        "required_prefix_patterns_before_first_marker": [
            {
                "name": "usart2_trace_enable",
                "pattern": r"USART2\s*->\s*CR1\s*(?:\|=|=)[^;]*",
                "reason": "trace channel must be enabled before the first marker",
            }
        ],
        "required_patterns_between_markers": [
            {
                "after_marker": "A",
                "before_marker": "B",
                "name": "gpioa_odr_write_between_a_b",
                "pattern": r"GPIOA\s*->\s*ODR\s*(?:\|=|=)[^;]*(?:0x20|1[Uu]?\s*<<\s*5)",
                "reason": "the GPIO write must land between A and B",
            }
        ],
    }
    assert _source_static_gate_errors(source, ["A", "B", "DONE"], gate) == []


def test_trace_state_expect_satisfies_required_marker_scan() -> None:
    source = """
    #include <stdint.h>
    int main(void) {
        trace_state_expect("A", 0x40000000, 0x1, 0u);
        trace_state("B", 0x40000000, 0x1);
        trace("DONE");
        return 0;
    }
    """
    assert _missing_required_trace_markers(source, ["A", "B", "DONE"]) == []


def test_source_gate_accepts_trace_state_expect_markers() -> None:
    source = """
    #include <stdint.h>
    int main(void) {
        USART2->CR1 |= (1u << 3);
        trace_state_expect("A", 0x40000000, 0x1, 0u);
        GPIOA->ODR |= 0x20u;
        trace_state("B", 0x40020014, 0x20);
        trace("DONE");
        return 0;
    }
    """
    gate = {
        "mode": "mainline_ordered_logs_v1",
        "required_prefix_patterns_before_first_marker": [
            {
                "name": "usart2_trace_enable",
                "pattern": r"USART2\s*->\s*CR1\s*(?:\|=|=)[^;]*",
            }
        ],
        "required_patterns_between_markers": [
            {
                "after_marker": "A",
                "before_marker": "B",
                "name": "gpioa_odr_write_between_a_b",
                "pattern": r"GPIOA\s*->\s*ODR\s*(?:\|=|=)[^;]*(?:0x20|1[Uu]?\s*<<\s*5)",
            }
        ],
    }
    assert _source_static_gate_errors(source, ["A", "B", "DONE"], gate) == []


def test_between_marker_gate_rejects_pre_marker_write() -> None:
    source = """
    #include <stdint.h>
    void trace(const char *);
    int main(void) {
        GPIOA->ODR |= 0x20u;
        trace("A");
        trace("B");
        trace("DONE");
        return 0;
    }
    """
    gate = {
        "mode": "mainline_ordered_logs_v1",
        "source_gate_repair_budget": 2,
        "target_bound_state_anchor_required": True,
        "required_patterns_between_markers": [
            {
                "after_marker": "A",
                "before_marker": "B",
                "name": "gpioa_odr_write_between_a_b",
                "pattern": r"GPIOA\s*->\s*ODR\s*(?:\|=|=)[^;]*(?:0x20|1[Uu]?\s*<<\s*5)",
                "reason": "the GPIO write must land between A and B",
            }
        ],
    }
    errors = _source_static_gate_errors(source, ["A", "B", "DONE"], gate)
    assert "source_static_gate_missing_between_marker_pattern:A:B:gpioa_odr_write_between_a_b" in errors


def test_source_gate_repair_budget_can_be_reduced_to_zero() -> None:
    assert _source_gate_repair_budget({"source_gate_repair_budget": 0}) == 0
    assert _source_gate_repair_budget({"source_gate_repair_budget": 3}) == 3
