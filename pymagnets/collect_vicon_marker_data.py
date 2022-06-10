
import pyrealtime as prt

from vicon_lib import get_vicon_source



def main():
    vicon_data = get_vicon_source()
    prt.RecordLayer(vicon_data.get_port("markers"), file_prefix="markers")

    prt.LayerManager.session().run()


if __name__=="__main__":
    main()