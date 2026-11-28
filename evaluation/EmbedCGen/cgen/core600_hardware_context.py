"""Core600 benchmark hardware context fragments.

This module owns the large static C context used by the Core600 benchmark
runner. Prompt contract fragments live in :mod:`cgen.hardware_contract`; this
module is intentionally limited to C context snippets and requirement-based
context selection.
"""

from __future__ import annotations

import re


BASE_CONTEXT = r"""#include <stdint.h>
#include <stddef.h>

__attribute__((weak)) void *memcpy(void *dest, const void *src, size_t n) {
    uint8_t *d = (uint8_t *)dest;
    const uint8_t *s = (const uint8_t *)src;
    for (size_t i = 0; i < n; ++i) {
        d[i] = s[i];
    }
    return dest;
}

__attribute__((weak)) void *memmove(void *dest, const void *src, size_t n) {
    uint8_t *d = (uint8_t *)dest;
    const uint8_t *s = (const uint8_t *)src;
    if (d < s) {
        for (size_t i = 0; i < n; ++i) {
            d[i] = s[i];
        }
    } else if (d > s) {
        for (size_t i = n; i > 0; --i) {
            d[i - 1] = s[i - 1];
        }
    }
    return dest;
}

__attribute__((weak)) void *memset(void *dest, int value, size_t n) {
    uint8_t *d = (uint8_t *)dest;
    for (size_t i = 0; i < n; ++i) {
        d[i] = (uint8_t)value;
    }
    return dest;
}

static inline void SystemInit(void) {}

// --- INFRASTRUCTURE (Always Available) ---
typedef struct { volatile uint32_t CR; volatile uint32_t PLLCFGR; volatile uint32_t CFGR; volatile uint32_t CIR; volatile uint32_t AHB1RSTR; volatile uint32_t AHB2RSTR; volatile uint32_t AHB3RSTR; uint32_t _reserved0; volatile uint32_t APB1RSTR; volatile uint32_t APB2RSTR; uint32_t _reserved1[2]; volatile uint32_t AHB1ENR; volatile uint32_t AHB2ENR; volatile uint32_t AHB3ENR; uint32_t _reserved2; volatile uint32_t APB1ENR; volatile uint32_t APB2ENR; uint32_t _reserved_rcc[12]; volatile uint32_t CSR; } RCC_TypeDef;
#define RCC ((RCC_TypeDef *) 0x40023800)

typedef struct {
    volatile uint32_t MODER; volatile uint32_t OTYPER; volatile uint32_t OSPEEDR; volatile uint32_t PUPDR;
    volatile uint32_t IDR; volatile uint32_t ODR; volatile uint32_t BSRR; volatile uint32_t LCKR;
    union {
        volatile uint32_t AFR[2];
        struct { volatile uint32_t AFRL; volatile uint32_t AFRH; };
    };
} GPIO_TypeDef;
#define GPIOA ((GPIO_TypeDef *) 0x40020000)
#define GPIOB ((GPIO_TypeDef *) 0x40020400)
#define GPIOC ((GPIO_TypeDef *) 0x40020800)
#define GPIOD ((GPIO_TypeDef *) 0x40020C00)
#define GPIOE ((GPIO_TypeDef *) 0x40021000)
#define GPIOF ((GPIO_TypeDef *) 0x40021400)
#define GPIOG ((GPIO_TypeDef *) 0x40021800)
#define GPIOH ((GPIO_TypeDef *) 0x40021C00)
#define GPIOI ((GPIO_TypeDef *) 0x40022000)

typedef struct {
    union {
        volatile uint32_t ISER[8];
        struct {
            volatile uint32_t ISER0; volatile uint32_t ISER1; volatile uint32_t ISER2; volatile uint32_t ISER3;
            volatile uint32_t ISER4; volatile uint32_t ISER5; volatile uint32_t ISER6; volatile uint32_t ISER7;
        };
    };
    uint32_t RESERVED0[24];
    volatile uint32_t ICER[8];
    uint32_t RESERVED1[24];
    volatile uint32_t ISPR[8];
    uint32_t RESERVED2[24];
    volatile uint32_t ICPR[8];
    uint32_t RESERVED3[24];
    volatile uint32_t IABR[8];
    uint32_t RESERVED4[56];
    union {
        volatile uint32_t IPR[60];
        volatile uint8_t IP[240];
    };
    uint32_t RESERVED5[644];
    volatile uint32_t STIR;
} NVIC_TypeDef;
#define NVIC ((NVIC_TypeDef *) 0xE000E100)
#define NVIC_IPR0  (*(volatile uint32_t *)0xE000E400)
#define NVIC_IPR1  (*(volatile uint32_t *)0xE000E404)
#define NVIC_IPR2  (*(volatile uint32_t *)0xE000E408)
#define NVIC_IPR3  (*(volatile uint32_t *)0xE000E40C)
#define NVIC_IPR4  (*(volatile uint32_t *)0xE000E410)
#define NVIC_IPR5  (*(volatile uint32_t *)0xE000E414)
#define NVIC_IPR6  (*(volatile uint32_t *)0xE000E418)
#define NVIC_IPR7  (*(volatile uint32_t *)0xE000E41C)
#define NVIC_IPR8  (*(volatile uint32_t *)0xE000E420)
#define NVIC_IPR9  (*(volatile uint32_t *)0xE000E424)
#define NVIC_IPR10 (*(volatile uint32_t *)0xE000E428)
#define NVIC_IPR11 (*(volatile uint32_t *)0xE000E42C)
#define NVIC_IPR12 (*(volatile uint32_t *)0xE000E430)
#define NVIC_IPR13 (*(volatile uint32_t *)0xE000E434)
#define NVIC_IPR14 (*(volatile uint32_t *)0xE000E438)
#define NVIC_IPR15 (*(volatile uint32_t *)0xE000E43C)
#define NVIC_IPR16 (*(volatile uint32_t *)0xE000E440)
#define NVIC_IPR17 (*(volatile uint32_t *)0xE000E444)
#define NVIC_IPR18 (*(volatile uint32_t *)0xE000E448)
#define NVIC_IPR19 (*(volatile uint32_t *)0xE000E44C)
#define NVIC_IPR20 (*(volatile uint32_t *)0xE000E450)

// --- SCB (System Control Block) ---
typedef struct {
    volatile uint32_t CPUID; volatile uint32_t ICSR; volatile uint32_t VTOR; volatile uint32_t AIRCR;
    volatile uint32_t SCR; volatile uint32_t CCR;
    union {
        struct { volatile uint32_t SHPR1; volatile uint32_t SHPR2; volatile uint32_t SHPR3; };
        volatile uint8_t SHP[12];
    };
    volatile uint32_t SHCSR; volatile uint32_t CFSR; volatile uint32_t HFSR; volatile uint32_t DFSR;
    volatile uint32_t MMFAR; volatile uint32_t BFAR; volatile uint32_t AFSR; volatile uint32_t ID_PFR[2];
    volatile uint32_t ID_DFR; volatile uint32_t ID_AFR; volatile uint32_t ID_MFR[4]; volatile uint32_t ID_ISAR[5];
    uint32_t RESERVED0[1]; volatile uint32_t CLIDR; volatile uint32_t CTR; volatile uint32_t CCSIDR;
    volatile uint32_t CSSELR; uint32_t RESERVED1[2]; volatile uint32_t CPACR;
} SCB_TypeDef;
#define SCB ((SCB_TypeDef *) 0xE000ED00)

// --- IRQn enumeration (STM32F407) ---
typedef enum {
    NonMaskableInt_IRQn = -14, HardFault_IRQn = -13, MemoryManagement_IRQn = -12,
    BusFault_IRQn = -11, UsageFault_IRQn = -10, SVCall_IRQn = -5,
    DebugMonitor_IRQn = -4, PendSV_IRQn = -2, SysTick_IRQn = -1,
    WWDG_IRQn = 0, PVD_IRQn = 1, TAMP_STAMP_IRQn = 2, RTC_WKUP_IRQn = 3,
    FLASH_IRQn = 4, RCC_IRQn = 5, EXTI0_IRQn = 6, EXTI1_IRQn = 7,
    EXTI2_IRQn = 8, EXTI3_IRQn = 9, EXTI4_IRQn = 10, DMA1_Stream0_IRQn = 11,
    DMA1_Stream1_IRQn = 12, DMA1_Stream2_IRQn = 13, DMA1_Stream3_IRQn = 14,
    DMA1_Stream4_IRQn = 15, DMA1_Stream5_IRQn = 16, DMA1_Stream6_IRQn = 17,
    ADC_IRQn = 18, CAN1_TX_IRQn = 19, CAN1_RX0_IRQn = 20, CAN1_RX1_IRQn = 21,
    CAN1_SCE_IRQn = 22, EXTI9_5_IRQn = 23, TIM1_BRK_TIM9_IRQn = 24,
    TIM1_UP_TIM10_IRQn = 25, TIM1_TRG_COM_TIM11_IRQn = 26, TIM1_CC_IRQn = 27,
    TIM2_IRQn = 28, TIM3_IRQn = 29, TIM4_IRQn = 30, I2C1_EV_IRQn = 31,
    I2C1_ER_IRQn = 32, I2C2_EV_IRQn = 33, I2C2_ER_IRQn = 34, SPI1_IRQn = 35,
    SPI2_IRQn = 36, USART1_IRQn = 37, USART2_IRQn = 38, USART3_IRQn = 39,
    EXTI15_10_IRQn = 40, RTC_Alarm_IRQn = 41, OTG_FS_WKUP_IRQn = 42,
    TIM8_BRK_TIM12_IRQn = 43, TIM8_UP_TIM13_IRQn = 44, TIM8_TRG_COM_TIM14_IRQn = 45,
    TIM8_CC_IRQn = 46, DMA1_Stream7_IRQn = 47, FMC_IRQn = 48, SDIO_IRQn = 49,
    TIM5_IRQn = 50, SPI3_IRQn = 51, UART4_IRQn = 52, UART5_IRQn = 53,
    TIM6_DAC_IRQn = 54, TIM7_IRQn = 55, DMA2_Stream0_IRQn = 56,
    DMA2_Stream1_IRQn = 57, DMA2_Stream2_IRQn = 58, DMA2_Stream3_IRQn = 59,
    DMA2_Stream4_IRQn = 60, ETH_IRQn = 61, ETH_WKUP_IRQn = 62, CAN2_TX_IRQn = 63,
    CAN2_RX0_IRQn = 64, CAN2_RX1_IRQn = 65, CAN2_SCE_IRQn = 66, OTG_FS_IRQn = 67,
    DMA2_Stream5_IRQn = 68, DMA2_Stream6_IRQn = 69, DMA2_Stream7_IRQn = 70,
    USART6_IRQn = 71, I2C3_EV_IRQn = 72, I2C3_ER_IRQn = 73, OTG_HS_EP1_OUT_IRQn = 74,
    OTG_HS_EP1_IN_IRQn = 75, OTG_HS_WKUP_IRQn = 76, OTG_HS_IRQn = 77,
    DCMI_IRQn = 78, CRYP_IRQn = 79, HASH_RNG_IRQn = 80, FPU_IRQn = 81,
} IRQn_Type;
// EXTI line IRQn aliases (shared interrupt lines)
#define EXTI5_IRQn    EXTI9_5_IRQn
#define EXTI6_IRQn    EXTI9_5_IRQn
#define EXTI7_IRQn    EXTI9_5_IRQn
#define EXTI8_IRQn    EXTI9_5_IRQn
#define EXTI9_IRQn    EXTI9_5_IRQn
#define EXTI10_IRQn   EXTI15_10_IRQn
#define EXTI11_IRQn   EXTI15_10_IRQn
#define EXTI12_IRQn   EXTI15_10_IRQn
#define EXTI13_IRQn   EXTI15_10_IRQn
#define EXTI14_IRQn   EXTI15_10_IRQn
#define EXTI15_IRQn   EXTI15_10_IRQn

// --- ARM CMSIS Intrinsic Stubs (Cortex-M4) ---
#define __NOP()   __asm volatile("nop")
#define __WFI()   __asm volatile("wfi")
#define __WFE()   __asm volatile("wfe")
#define __SEV()   __asm volatile("sev")
#define __DSB()   __asm volatile("dsb" ::: "memory")
#define __ISB()   __asm volatile("isb" ::: "memory")
#define __DMB()   __asm volatile("dmb" ::: "memory")
static inline void __enable_irq(void)  { __asm volatile("cpsie i" ::: "memory"); }
static inline void __disable_irq(void) { __asm volatile("cpsid i" ::: "memory"); }
static inline uint32_t __get_PRIMASK(void)  { uint32_t r; __asm volatile("mrs %0, primask" : "=r"(r)); return r; }
static inline void __set_PRIMASK(uint32_t v) { __asm volatile("msr primask, %0" :: "r"(v) : "memory"); }
static inline uint32_t __get_BASEPRI(void)  { uint32_t r; __asm volatile("mrs %0, basepri" : "=r"(r)); return r; }
static inline void __set_BASEPRI(uint32_t v) { __asm volatile("msr basepri, %0" :: "r"(v) : "memory"); }
static inline uint32_t __get_FAULTMASK(void)  { uint32_t r; __asm volatile("mrs %0, faultmask" : "=r"(r)); return r; }
static inline void __set_FAULTMASK(uint32_t v) { __asm volatile("msr faultmask, %0" :: "r"(v) : "memory"); }
#define __REV(x)    __builtin_bswap32(x)
#define __REV16(x)  __builtin_bswap16(x)
#define __ROR(x,n)  (((x)>>(n))|((x)<<(32-(n))))
static inline uint32_t __LDREXW(volatile uint32_t *addr) { uint32_t r; __asm volatile("ldrex %0, [%1]" : "=r"(r) : "r"(addr)); return r; }
static inline uint32_t __STREXW(uint32_t v, volatile uint32_t *addr) { uint32_t r; __asm volatile("strex %0, %2, [%1]" : "=r"(r) : "r"(addr), "r"(v)); return r; }
static inline uint16_t __LDREXH(volatile uint16_t *addr) { uint16_t r; __asm volatile("ldrexh %0, [%1]" : "=r"(r) : "r"(addr)); return r; }
static inline uint32_t __STREXH(uint16_t v, volatile uint16_t *addr) { uint32_t r; __asm volatile("strexh %0, %2, [%1]" : "=r"(r) : "r"(addr), "r"(v)); return r; }
static inline uint8_t __LDREXB(volatile uint8_t *addr) { uint8_t r; __asm volatile("ldrexb %0, [%1]" : "=r"(r) : "r"(addr)); return r; }
static inline uint32_t __STREXB(uint8_t v, volatile uint8_t *addr) { uint32_t r; __asm volatile("strexb %0, %2, [%1]" : "=r"(r) : "r"(addr), "r"(v)); return r; }
#define __CLZ(x)    __builtin_clz(x)
#define __SSAT(x,n) ((x))
#define __USAT(x,n) ((x))
#define __CLREX()   __asm volatile("clrex")
#define __NVIC_PRIO_BITS  4
static inline void NVIC_EnableIRQ(IRQn_Type IRQn) { NVIC->ISER[((uint32_t)(IRQn) >> 5)] = (1 << ((uint32_t)(IRQn) & 0x1F)); }
static inline void NVIC_DisableIRQ(IRQn_Type IRQn) { NVIC->ICER[((uint32_t)(IRQn) >> 5)] = (1 << ((uint32_t)(IRQn) & 0x1F)); }
static inline void NVIC_SetPriority(IRQn_Type IRQn, uint32_t priority) { NVIC->IP[(uint32_t)(IRQn)] = (uint8_t)((priority << (8 - __NVIC_PRIO_BITS)) & 0xFF); }

// --- SYSTICK DEFINITIONS ---
typedef struct { volatile uint32_t CTRL; volatile uint32_t LOAD; volatile uint32_t VAL; volatile uint32_t CALIB; } STK_TypeDef;
#define STK ((STK_TypeDef *) 0xE000E010)
#define SysTick STK
// ISR: SysTick_Handler

// --- FLASH DEFINITIONS ---
typedef struct { volatile uint32_t ACR; volatile uint32_t KEYR; volatile uint32_t OPTKEYR; volatile uint32_t SR; volatile uint32_t CR; volatile uint32_t OPTCR; } FLASH_TypeDef;
#define FLASH ((FLASH_TypeDef *) 0x40023C00)

// --- EXTI / SYSCFG DEFINITIONS ---
typedef struct { volatile uint32_t IMR; volatile uint32_t EMR; volatile uint32_t RTSR; volatile uint32_t FTSR; volatile uint32_t SWIER; volatile uint32_t PR; } EXTI_TypeDef;
typedef struct { volatile uint32_t MEMRMP; volatile uint32_t PMC; volatile uint32_t EXTICR[4]; uint32_t RESERVED[2]; volatile uint32_t CMPCR; } SYSCFG_TypeDef;
#define EXTI ((EXTI_TypeDef *) 0x40013C00)
#define SYSCFG ((SYSCFG_TypeDef *) 0x40013800)
// --- EXTI Interrupt Mask Register bit masks ---
#define EXTI_IMR_IM0   (1 << 0)
#define EXTI_IMR_IM1   (1 << 1)
#define EXTI_IMR_IM2   (1 << 2)
#define EXTI_IMR_IM3   (1 << 3)
#define EXTI_IMR_IM4   (1 << 4)
#define EXTI_IMR_IM5   (1 << 5)
#define EXTI_IMR_IM6   (1 << 6)
#define EXTI_IMR_IM7   (1 << 7)
#define EXTI_IMR_IM8   (1 << 8)
#define EXTI_IMR_IM9   (1 << 9)
#define EXTI_IMR_IM10  (1 << 10)
#define EXTI_IMR_IM11  (1 << 11)
#define EXTI_IMR_IM12  (1 << 12)
#define EXTI_IMR_IM13  (1 << 13)
#define EXTI_IMR_IM14  (1 << 14)
#define EXTI_IMR_IM15  (1 << 15)
#define EXTI_IMR_IM16  (1 << 16)
#define EXTI_IMR_IM17  (1 << 17)
#define EXTI_IMR_IM18  (1 << 18)
#define EXTI_IMR_IM19  (1 << 19)
#define EXTI_IMR_IM20  (1 << 20)
#define EXTI_IMR_IM21  (1 << 21)
#define EXTI_IMR_IM22  (1 << 22)
#define EXTI_IMR_IM23  (1 << 23)
#define EXTI_RTSR_TR0  (1 << 0)
#define EXTI_RTSR_TR1  (1 << 1)
#define EXTI_RTSR_TR2  (1 << 2)
#define EXTI_RTSR_TR3  (1 << 3)
#define EXTI_RTSR_TR4  (1 << 4)
#define EXTI_RTSR_TR5  (1 << 5)
#define EXTI_RTSR_TR6  (1 << 6)
#define EXTI_RTSR_TR7  (1 << 7)
#define EXTI_RTSR_TR8  (1 << 8)
#define EXTI_RTSR_TR9  (1 << 9)
#define EXTI_RTSR_TR10 (1 << 10)
#define EXTI_RTSR_TR11 (1 << 11)
#define EXTI_RTSR_TR12 (1 << 12)
#define EXTI_RTSR_TR13 (1 << 13)
#define EXTI_RTSR_TR14 (1 << 14)
#define EXTI_RTSR_TR15 (1 << 15)
#define EXTI_FTSR_TR0  (1 << 0)
#define EXTI_FTSR_TR1  (1 << 1)
#define EXTI_FTSR_TR2  (1 << 2)
#define EXTI_FTSR_TR3  (1 << 3)
#define EXTI_FTSR_TR4  (1 << 4)
#define EXTI_FTSR_TR5  (1 << 5)
#define EXTI_FTSR_TR6  (1 << 6)
#define EXTI_FTSR_TR7  (1 << 7)
#define EXTI_FTSR_TR8  (1 << 8)
#define EXTI_FTSR_TR9  (1 << 9)
#define EXTI_FTSR_TR10 (1 << 10)
#define EXTI_FTSR_TR11 (1 << 11)
#define EXTI_FTSR_TR12 (1 << 12)
#define EXTI_FTSR_TR13 (1 << 13)
#define EXTI_FTSR_TR14 (1 << 14)
#define EXTI_FTSR_TR15 (1 << 15)
#define EXTI_PR_PR0    (1 << 0)
#define EXTI_PR_PR1    (1 << 1)
#define EXTI_PR_PR2    (1 << 2)
#define EXTI_PR_PR3    (1 << 3)
#define EXTI_PR_PR4    (1 << 4)
#define EXTI_PR_PR5    (1 << 5)
#define EXTI_PR_PR6    (1 << 6)
#define EXTI_PR_PR7    (1 << 7)
#define EXTI_PR_PR8    (1 << 8)
#define EXTI_PR_PR9    (1 << 9)
#define EXTI_PR_PR10  (1 << 10)
#define EXTI_PR_PR11  (1 << 11)
#define EXTI_PR_PR12  (1 << 12)
#define EXTI_PR_PR13  (1 << 13)
#define EXTI_PR_PR14  (1 << 14)
#define EXTI_PR_PR15  (1 << 15)
// --- EXTI Software Interrupt Event Register bit masks ---
#define EXTI_SWIER_SWI0   (1 << 0)
#define EXTI_SWIER_SWI1   (1 << 1)
#define EXTI_SWIER_SWI2   (1 << 2)
#define EXTI_SWIER_SWI3   (1 << 3)
#define EXTI_SWIER_SWI4   (1 << 4)
#define EXTI_SWIER_SWI5   (1 << 5)
#define EXTI_SWIER_SWI6   (1 << 6)
#define EXTI_SWIER_SWI7   (1 << 7)
#define EXTI_SWIER_SWI8   (1 << 8)
#define EXTI_SWIER_SWI9   (1 << 9)
#define EXTI_SWIER_SWI10  (1 << 10)
#define EXTI_SWIER_SWI11  (1 << 11)
#define EXTI_SWIER_SWI12  (1 << 12)
#define EXTI_SWIER_SWI13  (1 << 13)
#define EXTI_SWIER_SWI14  (1 << 14)
#define EXTI_SWIER_SWI15  (1 << 15)
#define EXTI_SWIER_SWI16  (1 << 16)
#define EXTI_SWIER_SWI17  (1 << 17)
#define EXTI_SWIER_SWI18  (1 << 18)
#define EXTI_SWIER_SWI19  (1 << 19)
#define EXTI_SWIER_SWI20  (1 << 20)
#define EXTI_SWIER_SWI21  (1 << 21)
#define EXTI_SWIER_SWI22  (1 << 22)
#define EXTI_SWIER_SWI23  (1 << 23)
// SWIER aliases (models use SWIER0 instead of SWI0)
#define EXTI_SWIER_SWIER0  EXTI_SWIER_SWI0
#define EXTI_SWIER_SWIER1  EXTI_SWIER_SWI1
#define EXTI_SWIER_SWIER2  EXTI_SWIER_SWI2
#define EXTI_SWIER_SWIER3  EXTI_SWIER_SWI3
#define EXTI_SWIER_SWIER4  EXTI_SWIER_SWI4
#define EXTI_SWIER_SWIER5  EXTI_SWIER_SWI5
#define EXTI_SWIER_SWIER6  EXTI_SWIER_SWI6
#define EXTI_SWIER_SWIER7  EXTI_SWIER_SWI7
#define EXTI_SWIER_SWIER8  EXTI_SWIER_SWI8
#define EXTI_SWIER_SWIER9  EXTI_SWIER_SWI9
#define EXTI_SWIER_SWIER10 EXTI_SWIER_SWI10
#define EXTI_SWIER_SWIER11 EXTI_SWIER_SWI11
#define EXTI_SWIER_SWIER12 EXTI_SWIER_SWI12
#define EXTI_SWIER_SWIER13 EXTI_SWIER_SWI13
#define EXTI_SWIER_SWIER14 EXTI_SWIER_SWI14
#define EXTI_SWIER_SWIER15 EXTI_SWIER_SWI15
#define EXTI_SWIER_SWIER16 EXTI_SWIER_SWI16
#define EXTI_SWIER_SWIER17 EXTI_SWIER_SWI17
#define EXTI_SWIER_SWIER18 EXTI_SWIER_SWI18
#define EXTI_SWIER_SWIER19 EXTI_SWIER_SWI19
#define EXTI_SWIER_SWIER20 EXTI_SWIER_SWI20
#define EXTI_SWIER_SWIER21 EXTI_SWIER_SWI21
#define EXTI_SWIER_SWIER22 EXTI_SWIER_SWI22
#define EXTI_SWIER_SWIER23 EXTI_SWIER_SWI23
// ISR: EXTI0_IRQHandler, EXTI1_IRQHandler, EXTI2_IRQHandler, EXTI3_IRQHandler, EXTI4_IRQHandler, EXTI9_5_IRQHandler, EXTI15_10_IRQHandler

// --- TIMER DEFINITIONS ---
typedef struct { volatile uint32_t CR1; volatile uint32_t CR2; volatile uint32_t SMCR; volatile uint32_t DIER; volatile uint32_t SR; volatile uint32_t EGR; volatile uint32_t CCMR1; volatile uint32_t CCMR2; volatile uint32_t CCER; volatile uint32_t CNT; volatile uint32_t PSC; volatile uint32_t ARR; volatile uint32_t CCR1; volatile uint32_t CCR2; volatile uint32_t CCR3; volatile uint32_t CCR4; } TIM_TypeDef;
typedef struct { volatile uint32_t CR1; volatile uint32_t CR2; volatile uint32_t SMCR; volatile uint32_t DIER; volatile uint32_t SR; volatile uint32_t EGR; volatile uint32_t CCMR1; volatile uint32_t CCMR2; volatile uint32_t CCER; volatile uint32_t CNT; volatile uint32_t PSC; volatile uint32_t ARR; volatile uint32_t RCR; volatile uint32_t CCR1; volatile uint32_t CCR2; volatile uint32_t CCR3; volatile uint32_t CCR4; volatile uint32_t BDTR; volatile uint32_t DCR; volatile uint32_t DMAR; } TIM_Advanced_TypeDef;
#define TIM1 ((TIM_Advanced_TypeDef *) 0x40010000)
#define TIM8 ((TIM_Advanced_TypeDef *) 0x40010400)
#define TIM2 ((TIM_TypeDef *) 0x40000000)
#define TIM3 ((TIM_TypeDef *) 0x40000400)
#define TIM4 ((TIM_TypeDef *) 0x40000800)
#define TIM5 ((TIM_TypeDef *) 0x40000C00)
#define TIM6 ((TIM_TypeDef *) 0x40001000)
#define TIM7 ((TIM_TypeDef *) 0x40001400)
// ISR: TIM1_UP_TIM10_IRQHandler, TIM1_TRG_COM_TIM11_IRQHandler, TIM1_CC_IRQHandler, TIM8_UP_TIM13_IRQHandler, TIM8_TRG_COM_TIM14_IRQHandler, TIM8_CC_IRQHandler, TIM2_IRQHandler, TIM3_IRQHandler, TIM4_IRQHandler, TIM5_IRQHandler, TIM7_IRQHandler
"""

OPTIONAL_SLICES = {
    "USART": r"""// --- USART DEFINITIONS ---
typedef struct { volatile uint32_t SR; volatile uint32_t DR; volatile uint32_t BRR; volatile uint32_t CR1; volatile uint32_t CR2; volatile uint32_t CR3; volatile uint32_t GTPR; } USART_TypeDef;
#define USART1 ((USART_TypeDef *) 0x40011000)
#define USART2 ((USART_TypeDef *) 0x40004400)
#define USART3 ((USART_TypeDef *) 0x40004800)
#define USART6 ((USART_TypeDef *) 0x40011400)
#define UART4 ((USART_TypeDef *) 0x40004C00)
#define UART5 ((USART_TypeDef *) 0x40005000)
#define USART4 UART4
#define USART5 UART5
// ISR: USART1_IRQHandler, USART2_IRQHandler, USART3_IRQHandler, USART6_IRQHandler
// --- USART Status Register bit masks ---
#define USART_SR_PE      (1 << 0)
#define USART_SR_FE      (1 << 1)
#define USART_SR_NF      (1 << 2)
#define USART_SR_NE      USART_SR_NF
#define USART_SR_ORE     (1 << 3)
#define USART_SR_IDLE    (1 << 4)
#define USART_SR_RXNE    (1 << 5)
#define USART_SR_TC      (1 << 6)
#define USART_SR_TXE     (1 << 7)
#define USART_SR_LBD     (1 << 8)
#define USART_SR_CTS     (1 << 9)
#define USART_CR1_UE     (1 << 13)
#define USART_CR1_TE     (1 << 3)
#define USART_CR1_RE     (1 << 2)
#define USART_CR1_RXNEIE (1 << 5)
#define USART_CR1_TCIE   (1 << 6)
#define USART_CR1_TXEIE  (1 << 7)
#define USART_CR3_DMAR   (1 << 6)
#define USART_CR3_DMAT   (1 << 7)
#define USART_CR1_IDLEIE (1 << 4)
""",
    "TIM": "",
    "DMA": r"""// --- DMA DEFINITIONS ---
typedef struct { volatile uint32_t LISR; volatile uint32_t HISR; volatile uint32_t LIFCR; volatile uint32_t HIFCR; } DMA_TypeDef;
typedef struct { volatile uint32_t CR; volatile uint32_t NDTR; volatile uint32_t PAR; volatile uint32_t M0AR; volatile uint32_t M1AR; volatile uint32_t FCR; } DMA_Stream_TypeDef;
#define DMA1 ((DMA_TypeDef *) 0x40026000)
#define DMA2 ((DMA_TypeDef *) 0x40026400)
#define DMA1_Stream0 ((DMA_Stream_TypeDef *) 0x40026010)
#define DMA1_Stream1 ((DMA_Stream_TypeDef *) 0x40026028)
#define DMA1_Stream2 ((DMA_Stream_TypeDef *) 0x40026040)
#define DMA1_Stream3 ((DMA_Stream_TypeDef *) 0x40026058)
#define DMA1_Stream4 ((DMA_Stream_TypeDef *) 0x40026070)
#define DMA1_Stream5 ((DMA_Stream_TypeDef *) 0x40026088)
#define DMA1_Stream6 ((DMA_Stream_TypeDef *) 0x400260A0)
#define DMA1_Stream7 ((DMA_Stream_TypeDef *) 0x400260B8)
#define DMA2_Stream0 ((DMA_Stream_TypeDef *) 0x40026410)
#define DMA2_Stream1 ((DMA_Stream_TypeDef *) 0x40026428)
#define DMA2_Stream2 ((DMA_Stream_TypeDef *) 0x40026440)
#define DMA2_Stream3 ((DMA_Stream_TypeDef *) 0x40026458)
#define DMA2_Stream4 ((DMA_Stream_TypeDef *) 0x40026470)
#define DMA2_Stream5 ((DMA_Stream_TypeDef *) 0x40026488)
#define DMA2_Stream6 ((DMA_Stream_TypeDef *) 0x400264A0)
#define DMA2_Stream7 ((DMA_Stream_TypeDef *) 0x400264B8)
// --- DMA interrupt flag bit masks ---
#define DMA_LISR_TCIF0  (1 << 5)    // Stream0 transfer complete
#define DMA_LISR_HTIF0  (1 << 4)    // Stream0 half transfer
#define DMA_LISR_TEIF0  (1 << 3)    // Stream0 transfer error
#define DMA_LISR_DMEIF0 (1 << 2)    // Stream0 direct mode error
#define DMA_LISR_FEIF0  (1 << 0)    // Stream0 FIFO error
#define DMA_LISR_TCIF1  (1 << 11)   // Stream1 transfer complete
#define DMA_LISR_HTIF1  (1 << 10)   // Stream1 half transfer
#define DMA_LISR_TEIF1  (1 << 9)    // Stream1 transfer error
#define DMA_LISR_DMEIF1 (1 << 8)    // Stream1 direct mode error
#define DMA_LISR_FEIF1  (1 << 6)    // Stream1 FIFO error
#define DMA_LISR_TCIF2  (1 << 21)
#define DMA_LISR_HTIF2  (1 << 20)
#define DMA_LISR_TEIF2  (1 << 19)
#define DMA_LISR_DMEIF2 (1 << 18)
#define DMA_LISR_FEIF2  (1 << 16)
#define DMA_LISR_TCIF3  (1 << 27)
#define DMA_LISR_HTIF3  (1 << 26)
#define DMA_LISR_TEIF3  (1 << 25)
#define DMA_LISR_DMEIF3 (1 << 24)
#define DMA_LISR_FEIF3  (1 << 22)
#define DMA_HISR_TCIF4  (1 << 5)
#define DMA_HISR_HTIF4  (1 << 4)
#define DMA_HISR_TEIF4  (1 << 3)
#define DMA_HISR_DMEIF4 (1 << 2)
#define DMA_HISR_FEIF4  (1 << 0)
#define DMA_HISR_TCIF5  (1 << 11)
#define DMA_HISR_HTIF5  (1 << 10)
#define DMA_HISR_TEIF5  (1 << 9)
#define DMA_HISR_DMEIF5 (1 << 8)
#define DMA_HISR_FEIF5  (1 << 6)
#define DMA_HISR_TCIF6  (1 << 21)
#define DMA_HISR_HTIF6  (1 << 20)
#define DMA_HISR_TEIF6  (1 << 19)
#define DMA_HISR_DMEIF6 (1 << 18)
#define DMA_HISR_FEIF6  (1 << 16)
#define DMA_HISR_TCIF7  (1 << 27)
#define DMA_HISR_HTIF7  (1 << 26)
#define DMA_HISR_TEIF7  (1 << 25)
#define DMA_HISR_DMEIF7 (1 << 24)
#define DMA_HISR_FEIF7  (1 << 22)
#define DMA_LIFCR_CTCIF0  (1 << 5)
#define DMA_LIFCR_CHTIF0  (1 << 4)
#define DMA_LIFCR_CTEIF0  (1 << 3)
#define DMA_LIFCR_CDMEIF0 (1 << 2)
#define DMA_LIFCR_CFEIF0  (1 << 0)
#define DMA_LIFCR_CTCIF1  (1 << 11)
#define DMA_LIFCR_CHTIF1  (1 << 10)
#define DMA_LIFCR_CTEIF1  (1 << 9)
#define DMA_LIFCR_CDMEIF1 (1 << 8)
#define DMA_LIFCR_CFEIF1  (1 << 6)
#define DMA_LIFCR_CTCIF2  (1 << 21)
#define DMA_LIFCR_CHTIF2  (1 << 20)
#define DMA_LIFCR_CTEIF2  (1 << 19)
#define DMA_LIFCR_CDMEIF2 (1 << 18)
#define DMA_LIFCR_CFEIF2  (1 << 16)
#define DMA_LIFCR_CTCIF3  (1 << 27)
#define DMA_LIFCR_CHTIF3  (1 << 26)
#define DMA_LIFCR_CTEIF3  (1 << 25)
#define DMA_LIFCR_CDMEIF3 (1 << 24)
#define DMA_LIFCR_CFEIF3  (1 << 22)
#define DMA_HIFCR_CTCIF4  (1 << 5)
#define DMA_HIFCR_CHTIF4  (1 << 4)
#define DMA_HIFCR_CTEIF4  (1 << 3)
#define DMA_HIFCR_CDMEIF4 (1 << 2)
#define DMA_HIFCR_CFEIF4  (1 << 0)
#define DMA_HIFCR_CTCIF5  (1 << 11)
#define DMA_HIFCR_CHTIF5  (1 << 10)
#define DMA_HIFCR_CTEIF5  (1 << 9)
#define DMA_HIFCR_CDMEIF5 (1 << 8)
#define DMA_HIFCR_CFEIF5  (1 << 6)
#define DMA_HIFCR_CTCIF6  (1 << 21)
#define DMA_HIFCR_CHTIF6  (1 << 20)
#define DMA_HIFCR_CTEIF6  (1 << 19)
#define DMA_HIFCR_CDMEIF6 (1 << 18)
#define DMA_HIFCR_CFEIF6  (1 << 16)
#define DMA_HIFCR_CTCIF7  (1 << 27)
#define DMA_HIFCR_CHTIF7  (1 << 26)
#define DMA_HIFCR_CTEIF7  (1 << 25)
#define DMA_HIFCR_CDMEIF7 (1 << 24)
#define DMA_HIFCR_CFEIF7  (1 << 22)
// --- DMA HISR stream0-3 aliases (physical flags are in LISR, but models sometimes reference via HISR) ---
#define DMA_HISR_TCIF0  DMA_LISR_TCIF0
#define DMA_HISR_HTIF0  DMA_LISR_HTIF0
#define DMA_HISR_TEIF0  DMA_LISR_TEIF0
#define DMA_HISR_DMEIF0 DMA_LISR_DMEIF0
#define DMA_HISR_FEIF0  DMA_LISR_FEIF0
#define DMA_HISR_TCIF1  DMA_LISR_TCIF1
#define DMA_HISR_HTIF1  DMA_LISR_HTIF1
#define DMA_HISR_TEIF1  DMA_LISR_TEIF1
#define DMA_HISR_DMEIF1 DMA_LISR_DMEIF1
#define DMA_HISR_FEIF1  DMA_LISR_FEIF1
#define DMA_HISR_TCIF2  DMA_LISR_TCIF2
#define DMA_HISR_HTIF2  DMA_LISR_HTIF2
#define DMA_HISR_TEIF2  DMA_LISR_TEIF2
#define DMA_HISR_DMEIF2 DMA_LISR_DMEIF2
#define DMA_HISR_FEIF2  DMA_LISR_FEIF2
#define DMA_HISR_TCIF3  DMA_LISR_TCIF3
#define DMA_HISR_HTIF3  DMA_LISR_HTIF3
#define DMA_HISR_TEIF3  DMA_LISR_TEIF3
#define DMA_HISR_DMEIF3 DMA_LISR_DMEIF3
#define DMA_HISR_FEIF3  DMA_LISR_FEIF3
// --- DMA HIFCR stream0-3 aliases (physical clear bits are in LIFCR) ---
#define DMA_HIFCR_CTCIF0  DMA_LIFCR_CTCIF0
#define DMA_HIFCR_CHTIF0  DMA_LIFCR_CHTIF0
#define DMA_HIFCR_CTEIF0  DMA_LIFCR_CTEIF0
#define DMA_HIFCR_CDMEIF0 DMA_LIFCR_CDMEIF0
#define DMA_HIFCR_CFEIF0  DMA_LIFCR_CFEIF0
#define DMA_HIFCR_CTCIF1  DMA_LIFCR_CTCIF1
#define DMA_HIFCR_CHTIF1  DMA_LIFCR_CHTIF1
#define DMA_HIFCR_CTEIF1  DMA_LIFCR_CTEIF1
#define DMA_HIFCR_CDMEIF1 DMA_LIFCR_CDMEIF1
#define DMA_HIFCR_CFEIF1  DMA_LIFCR_CFEIF1
#define DMA_HIFCR_CTCIF2  DMA_LIFCR_CTCIF2
#define DMA_HIFCR_CHTIF2  DMA_LIFCR_CHTIF2
#define DMA_HIFCR_CTEIF2  DMA_LIFCR_CTEIF2
#define DMA_HIFCR_CDMEIF2 DMA_LIFCR_CDMEIF2
#define DMA_HIFCR_CFEIF2  DMA_LIFCR_CFEIF2
#define DMA_HIFCR_CTCIF3  DMA_LIFCR_CTCIF3
#define DMA_HIFCR_CHTIF3  DMA_LIFCR_CHTIF3
#define DMA_HIFCR_CTEIF3  DMA_LIFCR_CTEIF3
#define DMA_HIFCR_CDMEIF3 DMA_LIFCR_CDMEIF3
#define DMA_HIFCR_CFEIF3  DMA_LIFCR_CFEIF3
// --- DMA LISR/HISR aliases (models sometimes reference high-stream flags via LISR) ---
#define DMA_LISR_TCIF5  DMA_HISR_TCIF5
#define DMA_LISR_HTIF5  DMA_HISR_HTIF5
#define DMA_LISR_TEIF5  DMA_HISR_TEIF5
#define DMA_LISR_DMEIF5 DMA_HISR_DMEIF5
#define DMA_LISR_FEIF5  DMA_HISR_FEIF5
#define DMA_LISR_TCIF6  DMA_HISR_TCIF6
#define DMA_LISR_HTIF6  DMA_HISR_HTIF6
#define DMA_LISR_TEIF6  DMA_HISR_TEIF6
#define DMA_LISR_DMEIF6 DMA_HISR_DMEIF6
#define DMA_LISR_FEIF6  DMA_HISR_FEIF6
#define DMA_LISR_TCIF7  DMA_HISR_TCIF7
#define DMA_LISR_HTIF7  DMA_HISR_HTIF7
#define DMA_LISR_TEIF7  DMA_HISR_TEIF7
#define DMA_LISR_DMEIF7 DMA_HISR_DMEIF7
#define DMA_LISR_FEIF7  DMA_HISR_FEIF7
#define DMA_LIFCR_CTCIF5  DMA_HIFCR_CTCIF5
#define DMA_LIFCR_CHTIF5  DMA_HIFCR_CHTIF5
#define DMA_LIFCR_CTEIF5  DMA_HIFCR_CTEIF5
#define DMA_LIFCR_CDMEIF5 DMA_HIFCR_CDMEIF5
#define DMA_LIFCR_CFEIF5  DMA_HIFCR_CFEIF5
#define DMA_LIFCR_CTCIF6  DMA_HIFCR_CTCIF6
#define DMA_LIFCR_CHTIF6  DMA_HIFCR_CHTIF6
#define DMA_LIFCR_CTEIF6  DMA_HIFCR_CTEIF6
#define DMA_LIFCR_CDMEIF6 DMA_HIFCR_CDMEIF6
#define DMA_LIFCR_CFEIF6  DMA_HIFCR_CFEIF6
#define DMA_LIFCR_CTCIF7  DMA_HIFCR_CTCIF7
#define DMA_LIFCR_CHTIF7  DMA_HIFCR_CHTIF7
#define DMA_LIFCR_CTEIF7  DMA_HIFCR_CTEIF7
#define DMA_LIFCR_CDMEIF7 DMA_HIFCR_CDMEIF7
#define DMA_LIFCR_CFEIF7  DMA_HIFCR_CFEIF7
#define DMA_SxCR_EN     (1 << 0)
#define DMA_SxCR_TCIE   (1 << 4)
#define DMA_SxCR_HTIE   (1 << 3)
#define DMA_SxCR_TEIE   (1 << 2)
#define DMA_SxCR_DMEIE  (1 << 1)
#define DMA_SxCR_CIRC   (1 << 8)
#define DMA_SxCR_DIR_Msk (3 << 6)
#define DMA_SxCR_PINC   (1 << 9)
#define DMA_SxCR_MINC   (1 << 10)
#define DMA_SxCR_PSIZE_Msk (3 << 11)
#define DMA_SxCR_MSIZE_Msk (3 << 13)
#define DMA_SxCR_TCIF   DMA_SxCR_TCIE
#define DMA_SxCR_CHSEL_0 (1 << 25)
#define DMA_SxCR_CHSEL_1 (1 << 26)
#define DMA_SxCR_CHSEL_2 (1 << 27)
#define DMA_SxCR_FEIE    (1 << 7)
#define DMA_SxCR_PSIZE_0 (1 << 11)
#define DMA_SxCR_PSIZE_1 (1 << 12)
#define DMA_SxCR_MSIZE_0 (1 << 13)
#define DMA_SxCR_MSIZE_1 (1 << 14)
#define DMA_SxCR_DMAT    (1 << 21)
// --- DMA Stream x FIFO Control Register bit masks ---
#define DMA_SxFCR_FTH_0  (1 << 0)
#define DMA_SxFCR_FTH_1  (1 << 1)
#define DMA_SxFCR_DMDIS  (1 << 2)
#define DMA_SxFCR_FS     (0x1F << 3)
#define DMA_SxFCR_FEIE   (1 << 7)
// ISR: DMA1_Stream0_IRQHandler ... DMA2_Stream7_IRQHandler
""",
    "EXTI": "",
    "SYSTICK": "",
    "ADC": r"""// --- ADC DEFINITIONS ---
typedef struct { volatile uint32_t SR; volatile uint32_t CR1; volatile uint32_t CR2; volatile uint32_t SMPR1; volatile uint32_t SMPR2; volatile uint32_t JOFR1; volatile uint32_t JOFR2; volatile uint32_t JOFR3; volatile uint32_t JOFR4; volatile uint32_t HTR; volatile uint32_t LTR; volatile uint32_t SQR1; volatile uint32_t SQR2; volatile uint32_t SQR3; volatile uint32_t JSQR; volatile uint32_t JDR1; volatile uint32_t JDR2; volatile uint32_t JDR3; volatile uint32_t JDR4; volatile uint32_t DR; } ADC_TypeDef;
typedef struct { volatile uint32_t CSR; volatile uint32_t CCR; volatile uint32_t CDR; } ADC_Common_TypeDef;
#define ADC1 ((ADC_TypeDef *) 0x40012000)
#define ADC2 ((ADC_TypeDef *) 0x40012100)
#define ADC3 ((ADC_TypeDef *) 0x40012200)
#define ADC_COMMON ((ADC_Common_TypeDef *) 0x40012300)
// ISR: ADC_IRQHandler
""",
    "CAN": r"""// --- CAN DEFINITIONS ---
typedef struct { volatile uint32_t TIR; volatile uint32_t TDTR; volatile uint32_t TDLR; volatile uint32_t TDHR; } CAN_TxMailBox_TypeDef;
typedef struct { volatile uint32_t RIR; volatile uint32_t RDTR; volatile uint32_t RDLR; volatile uint32_t RDHR; } CAN_FIFOMailBox_TypeDef;
typedef struct { volatile uint32_t FR1; volatile uint32_t FR2; } CAN_FilterRegister_TypeDef;
typedef struct {
    volatile uint32_t MCR; volatile uint32_t MSR; volatile uint32_t TSR; volatile uint32_t RF0R; volatile uint32_t RF1R;
    volatile uint32_t IER; volatile uint32_t ESR; volatile uint32_t BTR;
    uint32_t RESERVED0[88];
    CAN_TxMailBox_TypeDef sTxMailBox[3];
    CAN_FIFOMailBox_TypeDef sFIFOMailBox[2];
    uint32_t RESERVED1[12];
    volatile uint32_t FMR; volatile uint32_t FM1R; uint32_t RESERVED2; volatile uint32_t FS1R; uint32_t RESERVED3;
    volatile uint32_t FFA1R; uint32_t RESERVED4; volatile uint32_t FA1R; uint32_t RESERVED5[8];
    CAN_FilterRegister_TypeDef sFilterRegister[28];
} CAN_TypeDef;
#define CAN1 ((CAN_TypeDef *) 0x40006400)
#define CAN2 ((CAN_TypeDef *) 0x40006800)
// ISR: CAN1_TX_IRQHandler, CAN1_RX0_IRQHandler, CAN1_RX1_IRQHandler, CAN1_SCE_IRQHandler
// ISR: CAN2_TX_IRQHandler, CAN2_RX0_IRQHandler, CAN2_RX1_IRQHandler, CAN2_SCE_IRQHandler
""",
    "SPI": r"""// --- SPI DEFINITIONS ---
typedef struct { volatile uint32_t CR1; volatile uint32_t CR2; volatile uint32_t SR; volatile uint32_t DR; volatile uint32_t CRCPR; volatile uint32_t RXCRCR; volatile uint32_t TXCRCR; volatile uint32_t I2SCFGR; volatile uint32_t I2SPR; } SPI_TypeDef;
#define SPI1 ((SPI_TypeDef *) 0x40013000)
#define SPI2 ((SPI_TypeDef *) 0x40003800)
#define SPI3 ((SPI_TypeDef *) 0x40003C00)
// ISR: SPI1_IRQHandler, SPI2_IRQHandler, SPI3_IRQHandler
""",
    "I2C": r"""// --- I2C DEFINITIONS ---
typedef struct { volatile uint32_t CR1; volatile uint32_t CR2; volatile uint32_t OAR1; volatile uint32_t OAR2; volatile uint32_t DR; volatile uint32_t SR1; volatile uint32_t SR2; volatile uint32_t CCR; volatile uint32_t TRISE; } I2C_TypeDef;
#define I2C1 ((I2C_TypeDef *) 0x40005400)
#define I2C2 ((I2C_TypeDef *) 0x40005800)
// ISR: I2C1_EV_IRQHandler, I2C1_ER_IRQHandler, I2C2_EV_IRQHandler, I2C2_ER_IRQHandler
""",
    "WWDG": r"""// --- WWDG / IWDG DEFINITIONS ---
typedef struct { volatile uint32_t CR; volatile uint32_t CFR; volatile uint32_t SR; } WWDG_TypeDef;
typedef struct { volatile uint32_t KR; volatile uint32_t PR; volatile uint32_t RLR; volatile uint32_t SR; } IWDG_TypeDef;
#define WWDG ((WWDG_TypeDef *) 0x40002C00)
#define IWDG ((IWDG_TypeDef *) 0x40003000)
// ISR: WWDG_IRQHandler
""",
    "CRC": r"""// --- CRC DEFINITIONS ---
typedef struct { volatile uint32_t DR; volatile uint32_t IDR; volatile uint32_t CR; } CRC_TypeDef;
#define CRC ((CRC_TypeDef *) 0x40023000)
""",
    "FLASH": "",
    "DAC": r"""// --- DAC DEFINITIONS ---
typedef struct {
    volatile uint32_t CR; volatile uint32_t SWTRIGR; volatile uint32_t DHR12R1; volatile uint32_t DHR12L1;
    volatile uint32_t DHR8R1; volatile uint32_t DHR12R2; volatile uint32_t DHR12L2; volatile uint32_t DHR8R2;
    volatile uint32_t DHR12RD; volatile uint32_t DHR12LD; volatile uint32_t DHR8RD; volatile uint32_t DOR1;
    volatile uint32_t DOR2; volatile uint32_t SR;
} DAC_TypeDef;
#define DAC ((DAC_TypeDef *) 0x40007400)
// ISR: TIM6_DAC_IRQHandler
""",
}


def build_hw_context(requirement: str, usart_force: bool = False) -> str:
    """Build pure C hardware context (structs + #defines) for a requirement."""
    lower_req = requirement.lower()

    def has_any(words):
        return any(w in lower_req for w in words)

    injected_parts = [BASE_CONTEXT]
    if has_any(["uart", "usart", "serial"]) or re.search(r"usart\d", lower_req) or usart_force:
        injected_parts.append(OPTIONAL_SLICES.get("USART", ""))
    if has_any(["tim", "timer", "pwm", "counter", "delay"]) or re.search(r"tim\d", lower_req):
        injected_parts.append(OPTIONAL_SLICES.get("TIM", ""))
    if has_any(["exti", "external interrupt", "gpio interrupt", "button", "rising", "falling", "interrupt", "irq", "isr"]):
        injected_parts.append(OPTIONAL_SLICES.get("EXTI", ""))
    if has_any(["systick", "tick", "delay_ms", "delay_us"]):
        injected_parts.append(OPTIONAL_SLICES.get("SYSTICK", ""))
    if has_any(["adc", "analog"]) or re.search(r"adc\d", lower_req):
        injected_parts.append(OPTIONAL_SLICES.get("ADC", ""))
    if has_any(["can", "bxcan"]) or re.search(r"can\d", lower_req):
        injected_parts.append(OPTIONAL_SLICES.get("CAN", ""))
    if has_any(["dac"]):
        injected_parts.append(OPTIONAL_SLICES.get("DAC", ""))
    if has_any(
        [
            "dma",
            "half-buffer",
            "half buffer",
            "two-half buffer",
            "double-buffer",
            "double buffer",
            "buffer handoff",
            "handoff buffer",
        ]
    ) or re.search(r"dma\d", lower_req):
        injected_parts.append(OPTIONAL_SLICES.get("DMA", ""))
    if has_any(["spi", "i2s"]) or re.search(r"spi\d", lower_req) or re.search(r"i2s\d", lower_req):
        injected_parts.append(OPTIONAL_SLICES.get("SPI", ""))
    if has_any(["i2c"]) or re.search(r"i2c\d", lower_req):
        injected_parts.append(OPTIONAL_SLICES.get("I2C", ""))
    if has_any(["wwdg", "iwdg", "watchdog", "window watchdog", "independent watchdog"]):
        injected_parts.append(OPTIONAL_SLICES.get("WWDG", ""))
    if has_any(["crc"]):
        injected_parts.append(OPTIONAL_SLICES.get("CRC", ""))
    if has_any(["flash", "flash erase", "flash program"]):
        injected_parts.append(OPTIONAL_SLICES.get("FLASH", ""))

    return "\n".join([ctx for ctx in injected_parts if ctx])
