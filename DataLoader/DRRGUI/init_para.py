from .init_layout import InitUILayout


class InitPara(InitUILayout):
    def __init__(self, ui_config, drr_config):
        super().__init__(ui_config)
        self.drr_config = drr_config
        self.projector = None
        self.drr_pixmap = None
        # 设置初始的投影参数
        self.d_s2p_text.setText(str(self.drr_config["d_s2p"]))
        self.im_sz_text.setText(str(self.drr_config["im_sz"]))