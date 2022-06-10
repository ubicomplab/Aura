import pyrealtime as prt

def main():
    in_slow = prt.InputLayer(rate=120)
    in_fast = prt.InputLayer(rate=217)

    merged = prt.MergeLayer(None, trigger=prt.LayerTrigger.SLOWEST, discard_old=True)
    merged.set_input(in_slow, 'in_slow')
    merged.set_input(in_fast, 'in_fast')
    prt.SplitLayer

    prt.PrintLayer(merged)

    prt.LayerManager.session().start()

if __name__=="__main__":
    main()