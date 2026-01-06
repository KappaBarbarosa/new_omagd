REGISTRY = {}

from .hpn_controller import HPNMAC
from .basic_controller import BasicMAC
from .n_controller import NMAC
from .n_graph_controller import NGraphMAC
from .updet_controller import UPDETController

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["n_graph_mac"] = NGraphMAC
REGISTRY["hpn_mac"] = HPNMAC
REGISTRY["updet_mac"] = UPDETController
