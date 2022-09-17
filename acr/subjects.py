from . import utils as acu
from . import io as acio
import pandas as pd

a4_info = {}
a4_info["subject"] = "ACR_4"
a4_info["complete_key_list"] = ["laser1-bl", "laser1-sd"]
a4_info["paths"] = acio.get_acr_paths(a4_info["subject"], a4_info["complete_key_list"])
a4_info["times"] = {}

a4_info["load_times"] = {}
a4_info["load_times"]["laser1-bl"] = (0, 18000)
a4_info["load_times"]["laser1-sd"] = (0, 0)
a4_info["channels"] = {}
a4_info["channels"]["EEGr"] = [1, 2]
a4_info["channels"]["EMGr"] = [1, 2]

a9_info = {}
a9_info["subject"] = "ACR_9"
a9_info["complete_key_list"] = ["control1", "laser1"]
a9_info["paths"] = acio.get_acr_paths(a9_info["subject"], a9_info["complete_key_list"])
a9_info["times"] = {}

a9_times_c1 = {}
a9_times_c1["bl_sleep_start"] = 5753
a9_times_c1["stim_on"] = 6574
a9_times_c1["stim_off"] = 20974
a9_times_c1["stim_on_dt"] = pd.Timestamp("2022-06-14 10:40:27.383717400")
a9_times_c1["stim_off_dt"] = pd.Timestamp("2022-06-14 14:40:27.385437720")
a9_times_l1 = {}
a9_times_l1["bl_sleep_start"] = 5753
a9_times_l1["stim_on"] = 9921
a9_times_l1["stim_off"] = 24321
a9_times_l1["stim_on_dt"] = pd.Timestamp("2022-06-17 11:18:33.631271960")
a9_times_l1["stim_off_dt"] = pd.Timestamp("2022-06-17 15:18:33.632992280")

a9_info["times"]["control1"] = a9_times_c1
a9_info["times"]["laser1"] = a9_times_l1

a9_info["load_times"] = {}
a9_info["load_times"]["control1"] = (0, 0)
a9_info["load_times"]["laser1"] = (0, 0)

a9_info["channels"] = {}
a9_info["channels"]["EEGr"] = [1, 2]
a9_info["channels"]["NNXr"] = list(range(1, 17))
a9_info["channels"]["LFP_"] = list(range(1, 17))

a10_info = {}
a10_info["subject"] = "ACR_10"
a10_info["complete_key_list"] = ["laser1", "laser1-bl"]
a10_info["paths"] = acio.get_acr_paths(
    a10_info["subject"], a10_info["complete_key_list"]
)
a10_info["start_times"] = {}
a10_info["start_times"]["laser1"] = 4158
a10_info["start_times"]["laser1-bl"] = 0
a10_info["load_times"] = {}
a10_info["load_times"]["laser1"] = (0, 0)
a10_info["load_times"]["laser1-bl"] = (0, 0)
a10_info["channels"] = {}
a10_info["channels"]["EEGr"] = [1, 2]

a11_info = {}
a11_info["subject"] = "ACR_11"
a11_info["complete_key_list"] = ["laser1", "laser1-bl"]
a11_info["paths"] = acio.get_acr_paths(
    a11_info["subject"], a11_info["complete_key_list"]
)
a11_info["start_times"] = {}
a11_info["start_times"]["laser1"] = 5974
a11_info["start_times"]["laser1-bl"] = 0
a11_info["load_times"] = {}
a11_info["load_times"]["laser1"] = (0, 0)
a11_info["load_times"]["laser1-bl"] = (0, 0)
a11_info["channels"] = {}
a11_info["channels"]["EEGr"] = [1, 2]

a12_info = {}
a12_info["subject"] = "ACR_12"
a12_info["complete_key_list"] = ["control1-bl", "laser1-bl", "control1", "laser1"]
a12_info["paths"] = acio.get_acr_paths(
    a12_info["subject"], a12_info["complete_key_list"]
)
a12_info["load_times"] = {}
a12_info["load_times"]["control1-bl"] = (0, 0)
a12_info["load_times"]["laser1-bl"] = (0, 0)
a12_info["load_times"]["control1"] = (0, 0)
a12_info["load_times"]["laser1"] = (0, 0)
a12_info["channels"] = {}
a12_info["channels"]["EEGr"] = [1, 2]
a12_info["channels"]["NNXr"] = list(range(1, 17))
a12_info["channels"]["LFP_"] = list(range(1, 17))

a13_info = {}
a13_info["subject"] = "ACR_13"
a13_info["complete_key_list"] = ["laser1-bl", "laser1"]
a13_info["paths"] = acio.get_acr_paths(
    a13_info["subject"], a13_info["complete_key_list"]
)
a13_info["load_times"] = {}
a13_info["load_times"]["laser1-bl"] = (0, 14400)
a13_info["load_times"]["laser1"] = (14027, 0)
a13_info["channels"] = {}
a13_info["channels"]["EEGr"] = [1, 2]
a13_info["channels"]["NNXr"] = list(range(1, 17))
a13_info["channels"]["LFP_"] = list(range(1, 17))
a13_info["channels"]["LFPo"] = list(range(1, 17))
a13_info["channels"]["NNXo"] = list(range(1, 17))


a14_info = {}
a14_info["subject"] = "ACR_14"
a14_info["complete_key_list"] = ["laser1-bl", "laser1"]
a14_info["load_times"] = {}
a14_info["load_times"]["laser1-bl"] = (0, 0)
a14_info["load_times"]["laser1"] = (13999, 0)
