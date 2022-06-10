/*
 * adc.c
 *
 *  Created on: Oct 25, 2018
 *      Author: farshid
 */
#include <driverlib.h>
#include <msp430.h>

#define OVERSAMPLE SD24_OVERSAMPLE_256


void adc_init() {
    //setting up the SD24
    SD24_init(SD24_BASE, SD24_REF_EXTERNAL); // Select external REF

    SD24_initConverterAdvancedParam param_channel0 = {0};
    param_channel0.converter = SD24_CONVERTER_0; // Select converter
    param_channel0.conversionMode = SD24_CONTINUOUS_MODE; // Select continues mode
    param_channel0.groupEnable = SD24_GROUPED; // Grouped
    param_channel0.inputChannel = SD24_INPUT_CH_ANALOG; // Input from analog signal
    param_channel0.dataFormat = SD24_DATA_FORMAT_BINARY; // 2’s complement data format
    param_channel0.interruptDelay = SD24_FIRST_SAMPLE_INTERRUPT; // 4th sample causes interrupt
    param_channel0.oversampleRatio = OVERSAMPLE; // Oversampling ratio 256
    param_channel0.gain = SD24_GAIN_1; // Preamplifier gain x1
    SD24_initConverterAdvanced(SD24_BASE, &param_channel0);

    SD24_initConverterAdvancedParam param_channel1 = {0};
    param_channel1.converter = SD24_CONVERTER_1; // Select converter
    param_channel1.conversionMode = SD24_CONTINUOUS_MODE; // Select continues mode
    param_channel1.groupEnable = SD24_GROUPED; // Grouped
    param_channel1.inputChannel = SD24_INPUT_CH_ANALOG; // Input from analog signal
    param_channel1.dataFormat = SD24_DATA_FORMAT_BINARY; // 2’s complement data format
    param_channel1.interruptDelay = SD24_FIRST_SAMPLE_INTERRUPT; // 4th sample causes interrupt
    param_channel1.oversampleRatio = OVERSAMPLE; // Oversampling ratio 256
    param_channel1.gain = SD24_GAIN_1; // Preamplifier gain x1
    SD24_initConverterAdvanced(SD24_BASE, &param_channel1);

    SD24_initConverterAdvancedParam param_channel2 = {0};
    param_channel2.converter = SD24_CONVERTER_2; // Select converter
    param_channel2.conversionMode = SD24_CONTINUOUS_MODE; // Select continues mode
    param_channel2.groupEnable = SD24_NOT_GROUPED; // Not Grouped
    param_channel2.inputChannel = SD24_INPUT_CH_ANALOG; // Input from analog signal
    param_channel2.dataFormat = SD24_DATA_FORMAT_BINARY; // 2’s complement data format
    param_channel2.interruptDelay = SD24_FIRST_SAMPLE_INTERRUPT; // 4th sample causes interrupt
    param_channel2.oversampleRatio = OVERSAMPLE; // Oversampling ratio 256
    param_channel2.gain = SD24_GAIN_1; // Preamplifier gain x1
    SD24_initConverterAdvanced(SD24_BASE, &param_channel2);
}


