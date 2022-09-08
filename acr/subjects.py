from . import utils as acu
from . import io as acio

a10_info = {}
a10_info["subject"] = "ACR_10"
a10_info["complete_key_list"] = ["laser1", "laser1-bl"]
a10_info["paths"] = acio.get_acr_paths(
    a10_info["subject"], a10_info["complete_key_list"]
)
a10_info["start_times"] = {}
a10_info["start_times"]["laser1"] = 4158
a10_info["start_times"]["laser1-bl"] = 0


a11_info = {}
a11_info["subject"] = "ACR_11"
a11_info["complete_key_list"] = ["laser1", "laser1-bl"]
a11_info["paths"] = acio.get_acr_paths(
    a11_info["subject"], a11_info["complete_key_list"]
)
a11_info["start_times"] = {}
a11_info["start_times"]["laser1"] = 5974
a11_info["start_times"]["laser1-bl"] = 0


a12_info = {}
a12_info["subject"] = "ACR_12"
a12_info["complete_key_list"] = ["control1-bl", "laser1-bl", "control1", "laser1"]
a12_info["paths"] = acio.get_acr_paths(
    a12_info["subject"], a12_info["complete_key_list"]
)
a12_info["load_times"] = {}
a12_info["load_times"]["control1-bl"] = (0, 43200)
a12_info["load_times"]["laser1-bl"] = (0, 43200)
a12_info["load_times"]["control1"] = (0, 43200)
a12_info["load_times"]["laser1"] = (0, 43200)
a12_info["channels"] = {}
a12_info["channels"]["EEGr"] = [1, 2]
a12_info["channels"]["NNXr"] = list(range(1, 17))
a12_info["channels"]["LFP_"] = list(range(1, 17))
