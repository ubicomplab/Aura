/*
 * uart.c
 *
 *  Created on: Oct 31, 2018
 *      Author: farshid
 */
#include <msp430.h>
#include <driverlib.h>


void uart_init() {

    // Configuration for 115200 UART with SMCLK at 16384000
    // These values were generated using the online tool available at:
    // http://software-dl.ti.com/msp430/msp430 public sw/mcu/msp430/MSP430BaudRateConverter/index.html
    EUSCI_A_UART_initParam uartConfig = {
    EUSCI_A_UART_CLOCKSOURCE_SMCLK, // SMCLK Clock Source
    1, // BRDIV = 8  clockPrescalar
    0, // UCxBRF = 14  firstModReg
    0xF7, // UCxBRS = 34=0x22  secondModReg
    EUSCI_A_UART_NO_PARITY, // No Parity
    EUSCI_A_UART_LSB_FIRST, // MSB First
    EUSCI_A_UART_ONE_STOP_BIT, // One stop bit
    EUSCI_A_UART_MODE, // UART mode
    EUSCI_A_UART_OVERSAMPLING_BAUDRATE_GENERATION // Oversampling Baudrate
    };

    // Settings P1.2 and P1.3 as UART pins.
    GPIO_setAsPeripheralModuleFunctionInputPin(GPIO_PORT_P1,
    GPIO_PIN2 | GPIO_PIN3,
    GPIO_PRIMARY_MODULE_FUNCTION);

    // Configure and enable the UART peripheral
    EUSCI_A_UART_init(EUSCI_A0_BASE, &uartConfig);
    EUSCI_A_UART_enable(EUSCI_A0_BASE);
}


