import threading

from gui.window import OptionGui


def gui_thread(name: str):
    OptionGui()


gui_thread: threading.Thread = threading.Thread(target=gui_thread, args=(1,))
gui_thread.start()