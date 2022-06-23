from model_interface import ModelInterface
from configuration_manager import ConfigurationManager as gconfig
from basic_model import BasicModel
from fast_conv_segment import FastConvSegment
from dgcnn_segment import DgcnnSegment
from garnet_segment import GarNetSegment
from fcnn_segment import FcnnSegment
from gravnet_segment import GravnetSegment

class ModelFactory:
    def get_model(self):
        model = gconfig.get_config_param("model", "str")
        if model == "basic_conv_graph":
            model = BasicModel()
        elif model == "conv_graph_dgcnn_fast_conv":
            model = BasicModel()
            model.set_conv_segment(FastConvSegment())
            model.set_graph_segment(DgcnnSegment())
        elif model == "conv_graph_garnet_fast_conv":
            model = BasicModel()
            model.set_conv_segment(FastConvSegment())
            model.set_graph_segment(GarNetSegment())
        elif model == "conv_fcnn_fast_conv":
            model = BasicModel()
            model.set_conv_segment(FastConvSegment())
            model.set_graph_segment(FcnnSegment())
        elif model == "conv_grav_net_fast_conv":
            model = BasicModel()
            model.set_conv_segment(FastConvSegment())
            model.set_graph_segment(GravnetSegment())
        else:
            return ModelInterface() # TODO: Fix this

        return model
