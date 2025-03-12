import sys

import toml
from PyQt6.QtWidgets import QApplication

from DataLoader.DRRGUI.event_handler import EventHandler

class DRRInterface(EventHandler):
    def __init__(self, ui_config, drr_config):
        super().__init__(ui_config, drr_config)

if __name__ == '__main__':
    with open('config/drr_ui_config.toml', 'r', encoding='utf-8') as f:
        _ui_config = toml.load(f)
    with open('config/default_drr_para.toml', 'r', encoding='utf-8') as f:
        _drr_config = toml.load(f)
    app = QApplication(sys.argv)
    drr_gui = DRRInterface(_ui_config, _drr_config)
    drr_gui.show()
    sys.exit(app.exec())