import pyrealtime as prt
from controller_lib import get_device_data


def main():
    data = get_device_data()
    prt.LayerManager.session().run(show_monitor=False)


if __name__ == "__main__":
    main()
