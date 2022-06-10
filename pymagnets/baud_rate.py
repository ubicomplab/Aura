UCOS16 = 1
UCBRx = 1
UCBRFx = 1
UCBRSx = 0x92
# UCOS16 = 1
# UCBRx = 78
# UCBRFx = 2
# UCBRSx = 0x0

f_BRCLK = 16700000
# f_BRCLK = 12000000
TARGET_BAUD = 960000
# TARGET_BAUD = 9600

def main():
    m_UCBRSx = lambda i: 1 if UCBRSx & (0x01<<(7-(i%7))) else 0
    if UCOS16== 1:
        t_bit_i = lambda i: (1/f_BRCLK) * ((16 * UCBRx)+UCBRFx+m_UCBRSx(i))
    else:
        t_bit_i = lambda i: (1/f_BRCLK) * (UCBRx+m_UCBRSx(i))

    t_bit_end_i = lambda i: sum([t_bit_i(ii) for ii in range(i+1)])
    t_bit_end_ideal = lambda i: (1 / (TARGET_BAUD)) * (i+1)

    print(t_bit_end_i(9))
    print(t_bit_end_ideal(9))

    effective_baud = 1/(t_bit_end_i(9) / 10)
    target_baud = 1/(t_bit_end_ideal(9) / 10)
    print(effective_baud)
    print(target_baud)

    N = f_BRCLK / TARGET_BAUD
    if N > 16:
        print("OS16 = 1")
        print("UCBRx: ", int(N / 16))
        print("UCBRFx: ", int(N % 16))
        print("frac for UCBRSx: ", (N - int(N)) % 1)
    else:
        print("OS16 = 0")
        print("UCBRx: ", int(N))
        print("frac for UCBRSx: ", (N - int(N)) % 1)



if __name__ == '__main__':
    main()