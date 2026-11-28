"""Shared hardware-contract text for code generation prompts.

This module intentionally contains stable prompt contract fragments only. Large
generated C hardware contexts and benchmark-specific optional slices still live
next to their compilers/runners until they can be migrated safely.
"""

BENCHMARK_HARDWARE_CONTEXT_MEMBER_GUIDANCE = (
    "Use the exact member names from the provided hardware context; for benchmark contexts this is "
    "typically GPIOx->AFR[0]/AFR[1], NVIC->ISER[index], and NVIC->IPR[index]"
)

HARDWARE_CONTEXT_ALREADY_PROVIDED_GUIDANCE = (
    "The hardware context (SVD structs like RCC_TypeDef, GPIO_TypeDef) is ALREADY provided. "
    "NEVER redefine these structures or macros in your output, otherwise compilation will fail due to conflicts."
)

CMSIS_INLINE_ASM_GUIDANCE = (
    'Do NOT use __NOP(), __WFI() or similar CMSIS macros. Use inline assembly strictly: __asm__ volatile("nop");'
)

NO_HELPER_MACRO_INVENTION_GUIDANCE = (
    "Do NOT invent helper macros/types (e.g., GPIO_MODER_OUTPUT, GPIO_BSRR_SET, "
    "RCC_AHB1ENR_GPIOxEN) unless they exist in the provided context; use direct bit operations instead."
)

DMA_ARCHITECTURE_GUIDANCE = """DMA ARCHITECTURE (CRITICAL - models often get this wrong):
   STM32F4 DMA has TWO struct types with DIFFERENT register sets:
   - DMA_TypeDef (controller): LISR, HISR, LIFCR, HIFCR  (interrupt status/clear for ALL streams)
   - DMA_Stream_TypeDef (stream): CR, NDTR, PAR, M0AR, M1AR, FCR  (config for ONE stream)
   WRONG: DMA1_Stream5->LISR  (LISR belongs to DMA_TypeDef, NOT DMA_Stream_TypeDef)
   RIGHT: DMA1->LISR | DMA1->HISR | DMA1->LIFCR | DMA1->HIFCR
   RIGHT: DMA1_Stream5->CR | DMA1_Stream5->NDTR | DMA1_Stream5->PAR | DMA1_Stream5->M0AR
   Stream interrupt flags: LISR covers stream0-3, HISR covers stream4-7."""

PROMPT_CONTRACT_FRAGMENTS = {
    "benchmark_member_guidance": BENCHMARK_HARDWARE_CONTEXT_MEMBER_GUIDANCE,
    "hardware_context_already_provided": HARDWARE_CONTEXT_ALREADY_PROVIDED_GUIDANCE,
    "cmsis_inline_asm": CMSIS_INLINE_ASM_GUIDANCE,
    "no_helper_macro_invention": NO_HELPER_MACRO_INVENTION_GUIDANCE,
    "dma_architecture": DMA_ARCHITECTURE_GUIDANCE,
}
