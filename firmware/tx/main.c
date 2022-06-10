#include <msp430.h>
#include <stdint.h>

#define SWAP_UINT16(x) (((x) >> 8) | ((x) << 8))

// f = fMCLK/2^28 × FREQREG
// FREQREG = f * 2^28 / fMCLK
//1789570
// 20 kHz = (1789570)
#define F_MCLK 4000000
#define FREQREG (6710886) // 100 kHz
//#define FREQREG (7247757) // 108 kHz
#define FREQREG_LSB (FREQREG & 0x3FFF)
#define FREQREG_MSB ((FREQREG>>14) & 0x3FFF)

#define SQUARE 1
#define LOCK_CHANNEL_1 1
#define MS_PER_CHANNEL 3
#define PERIOD_MS 11

#if SQUARE==1
#define WAVEFORM (0x0068)
#else
#define WAVEFORM (0x0000)
#endif


static const union
{
    uint16_t tx_buffer[5];
    uint8_t tx_buffer_bytes[10];
} tx_data = {
             SWAP_UINT16(0x2100 | WAVEFORM),      //0010 0001 0110 1000
             SWAP_UINT16(0x4000 | FREQREG_LSB),   // FREQ0 LSB 0100 0011 0110 1010  Hex: 29D036A Dec 43844458
             SWAP_UINT16(0x4000 | FREQREG_MSB),   // FREQ0 MSB 0100 1010 0111 0100
             SWAP_UINT16(0xC000),
             SWAP_UINT16(0x2000 | WAVEFORM)
             };


uint8_t tx_index = 0;
uint8_t channel = 0;


void spi_init(){

    SYSCFG3 = 0x01;                           // Remap SPI to P1.1
    P1SEL0 |= BIT1 | BIT3;             // set 3-SPI pin as second function

    UCA0CTLW0 |= UCSWRST;                     // **Put state machine in reset**
    UCA0CTLW0 |= UCMST|UCSYNC|UCCKPL|UCMSB|UCCKPH;   // 3-pin, 8-bit SPI master
                                              // Clock polarity high, MSB
    UCA0CTLW0 |= UCSSEL__SMCLK;                // Select SMCLK
    UCA0BR0 = 0x02;                           // BRCLK = ACLK/2
    UCA0BR1 = 0;                              //
    UCA0MCTLW = 0;                            // No modulation
    UCA0CTLW0 &= ~UCSWRST;                    // **Initialize USCI state machine**


}



void program_dds(){

    P2OUT &= ~(0x01);

    UCA0IE |= UCTXIE;                     // Enable TX interrupt
}

void port_init(){
    P1SEL0 = 0;
    P1DIR = 0xFF;
    P1OUT = 0x00;

    P2SEL0 = 0;
    P2SEL1 = 0;

    P2DIR = 0xFF;
    P2OUT = 0x01;

}

void timer_init()
{
    TB0CCTL0 |= CCIE;                             // TBCCR0 interrupt enabled
    TB0CCR0 = 1000;
    TB0CTL = TBSSEL__SMCLK | MC__UP;             // SMCLK, UP mode
}

void clk_init()
{
    //__bis_SR_register(SCG0);                // disable FLL
   // CSCTL3 |= SELREF__REFOCLK;              // Set REFO as FLL reference source
    //CSCTL1 = DCOFTRIMEN_0 | DCOFTRIM | DCORSEL_0;
    //CSCTL2 = FLLD_0 + 243;                  // DCODIV = 8MHz
    //__delay_cycles(3);
    //__bic_SR_register(SCG0);                // enable FLL


    //CSCTL4 = SELMS__DCOCLKDIV | SELA__REFOCLK; // set default REFO(~32768Hz) as ACLK source, ACLK = 32768Hz
                                            // default DCODIV as MCLK and SMCLK source
    //CSCTL5 =

}

/**
 * main.c
 */
int main(void)
{
    WDTCTL = WDTPW | WDTHOLD;                 // Stop watchdog timer

    clk_init();

    port_init();
    spi_init();


    PM5CTL0 &= ~LOCKLPM5;                     // Disable the GPIO power-on default high-impedance mode
                                              // to activate previously configured port settings

    timer_init();

    program_dds();

    __bis_SR_register(LPM0_bits | GIE);   // enable global interrupts, enter LPM0

    while (1){
    }
	
	return 0;
}


#if defined(__TI_COMPILER_VERSION__) || defined(__IAR_SYSTEMS_ICC__)
#pragma vector=USCI_A0_VECTOR
__interrupt
#elif defined(__GNUC__)
__attribute__((interrupt(USCI_A0_VECTOR)))
#endif
void USCI_A0_ISR(void)
{
    switch(__even_in_range(UCA0IV, USCI_SPI_UCTXIFG))
    {
        case USCI_SPI_UCTXIFG:
            if (tx_index < sizeof(tx_data.tx_buffer_bytes)){
                UCA0TXBUF = tx_data.tx_buffer_bytes[tx_index++];
               // UCA0IE &= ~UCTXIE;
            }
            else
            {
                P2OUT |= (0x01);
                __bic_SR_register_on_exit(LPM0_bits);// Wake up
            }
            break;
        default:
            break;
    }
}

// Timer0_B0 interrupt service routine
#if defined(__TI_COMPILER_VERSION__) || defined(__IAR_SYSTEMS_ICC__)
#pragma vector = TIMER0_B0_VECTOR
__interrupt void Timer0_B0_ISR (void)
#elif defined(__GNUC__)
void __attribute__ ((interrupt(TIMER0_B0_VECTOR))) Timer0_B0_ISR (void)
#else
#error Compiler not supported!
#endif
{
    if (channel < MS_PER_CHANNEL)
    {
        P2OUT = (0x01);
    }
#if LOCK_CHANNEL_1 == 0
    else if (channel >= MS_PER_CHANNEL && channel < (MS_PER_CHANNEL*2))
    {
        P2OUT = (0x41);
    }
    else if (channel >= (MS_PER_CHANNEL*2) && channel < (MS_PER_CHANNEL*3))
    {
        P2OUT = (0x81);
    }
    else
    {
        P2OUT = (0xC1);
    }
#endif
    channel++;

    if (channel >= PERIOD_MS)
    {
        channel = 0;
    }
}
