/*
 * timer.c
 *
 *  Created on: Oct 31, 2018
 *      Author: farshid
 */
#include <driverlib.h>
#include <msp430.h>

#define PERIOD 128
//#define COMPARE_VALUE 100

void timer_init() {
    //Start TIMER A
//    Timer_A_initContinuousModeParam initContParam = {0};
//    initContParam.clockSource = TIMER_A_CLOCKSOURCE_SMCLK;
//    initContParam.clockSourceDivider = TIMER_A_CLOCKSOURCE_DIVIDER_1;
//    initContParam.timerInterruptEnable_TAIE = TIMER_A_TAIE_INTERRUPT_ENABLE;
//    initContParam.timerClear = TIMER_A_SKIP_CLEAR;
//    initContParam.startTimer = false;
//    Timer_A_initContinuousMode(TIMER_A1_BASE, &initContParam);

    //Initiaze up mode
//    Timer_A_clearCaptureCompareInterrupt(TIMER_A1_BASE,
//    TIMER_A_CAPTURECOMPARE_REGISTER_0
//    );
    Timer_A_initUpModeParam initUpParam = {0};
    initUpParam.clockSource = TIMER_A_CLOCKSOURCE_SMCLK;
    initUpParam.clockSourceDivider = TIMER_A_CLOCKSOURCE_DIVIDER_1;
    initUpParam.timerPeriod = PERIOD;
    initUpParam.timerInterruptEnable_TAIE = TIMER_A_TAIE_INTERRUPT_DISABLE;
    initUpParam.captureCompareInterruptEnable_CCR0_CCIE = TIMER_A_CCIE_CCR0_INTERRUPT_ENABLE;
    initUpParam.timerClear = TIMER_A_DO_CLEAR;
    initUpParam.startTimer = false;
    Timer_A_initUpMode(TIMER_A0_BASE, &initUpParam);
}


