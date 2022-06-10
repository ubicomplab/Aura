#include <msp430.h>
#include <driverlib.h>
#include <stdint.h>
#include <adc.h>
#include <uart.h>
#include <timer.h>
/**
 * main.c
 */
#define SIZE_OF_BUFFER_SIGN 50

bool ADC_DONE = false;
unsigned long results_channel0;
unsigned long results_channel1;
unsigned long results_channel2;

uint16_t sign_c0 = 0;
uint16_t sign_c1 = 0;
uint16_t sign_c2 = 0;
uint8_t buffer_index = 0;

uint8_t sign_byte = 0;
uint16_t packet_id = 0;

void detect_sign() {
//    int temp_sign = 0;
//    if (sign_c0 > buffer_index / 2) {
//        temp_sign =+ 0x01;
//    }
//    if (sign_c1 > buffer_index/2) {
//        temp_sign =+ 0x02;
//    }
//    if (sign_c2 > buffer_index/2) {
//        temp_sign = 0x04;
//    }
//    return temp_sign;
//    sign_byte = (((0x03 * sign_c0) / buffer_index) << 0) | (((0x03 * sign_c1) / buffer_index) << 2) | (((0x03 * sign_c2) / buffer_index) << 4);
    sign_byte = (((0x03 * sign_c0) >> 4) << 0) | (((0x03 * sign_c1) >> 4) << 2) | (((0x03 * sign_c2) >> 4) << 4);
}

int main(void)
{
    WDTCTL = WDTPW | WDTHOLD;   // stop watchdog timer

    // Set DCO Frequency to 16.384MHz
    CS_setupDCO(CS_EXTERNAL_RESISTOR);
    // Configure MCLK and SMCLK
    CS_initClockSignal(CS_MCLK, CS_CLOCK_DIVIDER_1);
    CS_initClockSignal(CS_SMCLK, CS_CLOCK_DIVIDER_1);

    //Setting up Uart
    uart_init();
    WDT_hold(WDT_BASE);

    //Setting up the SD24
    adc_init();

    //Setting up TIMER_A
    timer_init();

    //Setting P2.0, P2.1 and P2.2 as the coils sign
//    GPIO_setAsPeripheralModuleFunctionInputPin(GPIO_PORT_P2,
//    GPIO_PIN0 | GPIO_PIN1, GPIO_PIN2
//    GPIO_PRIMARY_MODULE_FUNCTION);
    GPIO_setAsInputPin(GPIO_PORT_P2, GPIO_PIN0);
    GPIO_setAsInputPin(GPIO_PORT_P2, GPIO_PIN1);
    GPIO_setAsInputPin(GPIO_PORT_P2, GPIO_PIN2);

    SD24_enableInterrupt(SD24_BASE, SD24_CONVERTER_2, SD24_CONVERTER_INTERRUPT);
    SD24_startConverterConversion(SD24_BASE, SD24_CONVERTER_2); // Set bit to start conversion
    Timer_A_startCounter(TIMER_A0_BASE, TIMER_A_UP_MODE);

	while (1)
	{
	    //Enter LPM0 w/ interrupts
	    __bis_SR_register(LPM0_bits | GIE);

	    if (ADC_DONE) {
	        detect_sign();


            EUSCI_A_UART_transmitData(EUSCI_A0_BASE, 'U');
            EUSCI_A_UART_transmitData(EUSCI_A0_BASE, 'W');

            EUSCI_A_UART_transmitData(EUSCI_A0_BASE, (packet_id & 0x00FF)>>0);
            EUSCI_A_UART_transmitData(EUSCI_A0_BASE, (packet_id & 0xFF00)>>8);
            EUSCI_A_UART_transmitData(EUSCI_A0_BASE, buffer_index);
            packet_id++;

            EUSCI_A_UART_transmitData(EUSCI_A0_BASE, ((results_channel0 & 0x0000FF00)>>8));
            EUSCI_A_UART_transmitData(EUSCI_A0_BASE, ((results_channel0 & 0x00FF0000)>>16));

            EUSCI_A_UART_transmitData(EUSCI_A0_BASE, ((results_channel1 & 0x0000FF00)>>8));
            EUSCI_A_UART_transmitData(EUSCI_A0_BASE, ((results_channel1 & 0x00FF0000)>>16));

            EUSCI_A_UART_transmitData(EUSCI_A0_BASE, ((results_channel2 & 0x0000FF00)>>8));
            EUSCI_A_UART_transmitData(EUSCI_A0_BASE, ((results_channel2 & 0x00FF0000)>>16));

            EUSCI_A_UART_transmitData(EUSCI_A0_BASE, sign_byte);

            buffer_index = 0;
            sign_c0 = 0;
            sign_c1 = 0;
            sign_c2 = 0;
            ADC_DONE = false;
            Timer_A_startCounter(TIMER_A0_BASE, TIMER_A_UP_MODE);
	    }
	}
	return 0;
}

//SD24 ISR
#if defined(__TI_COMPILER_VERSION__) || defined(__IAR_SYSTEMS_ICC__)
#pragma vector=SD24_VECTOR
__interrupt void SD24_ISR(void) {
#elif defined(__GNUC__)
void __attribute__ ((interrupt(SD24_VECTOR))) SD24_ISR (void) {
#else
#error Compiler not supported!
#endif
    switch (__even_in_range(SD24IV,SD24IV_SD24MEM3)) {
        case SD24IV_NONE: break;
        case SD24IV_SD24OVIFG: break;
        case SD24IV_SD24MEM0: break;
        case SD24IV_SD24MEM1: break;
        case SD24IV_SD24MEM2:
            Timer_A_stop(TIMER_A0_BASE);
            results_channel0 = SD24_getResults(SD24_BASE, SD24_CONVERTER_0);
            results_channel1 = SD24_getResults(SD24_BASE, SD24_CONVERTER_1);
            results_channel2 = SD24_getResults(SD24_BASE, SD24_CONVERTER_2);
            ADC_DONE = true;
            break;
        case SD24IV_SD24MEM3: break;
        default: break;
    }
    __bic_SR_register_on_exit(LPM0_bits);       // Wake up from LPM0
}
// TIMER_A ISR
#if defined(__TI_COMPILER_VERSION__) || defined(__IAR_SYSTEMS_ICC__)
#pragma vector=TIMER0_A0_VECTOR
__interrupt void TA0_ISR(void)
#elif defined(__GNUC__)
void __attribute__ ((interrupt(TIMER0_A0_VECTOR))) TA0_ISR (void)
#else
#error Compiler not supported!
#endif
{
    if(P2IN & BIT2)  sign_c0++;
    if(P2IN & BIT1)  sign_c1++;
    if(P2IN & BIT0)  sign_c2++;
//    sign_c0 += GPIO_getInputPinValue(GPIO_PORT_P2, GPIO_PIN0);
//    sign_c1 += GPIO_getInputPinValue(GPIO_PORT_P2, GPIO_PIN1);
//    sign_c2 += GPIO_getInputPinValue(GPIO_PORT_P2, GPIO_PIN2);
    buffer_index++;
//    __bic_SR_register_on_exit(LPM0_bits);       // Wake up from LPM0

}
