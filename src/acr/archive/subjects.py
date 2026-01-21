from .. import utils as acu
from .. import io as acio
import pandas as pd


def subject_info():
    """Reference for all keys of subject dictionaries

    Keys:
    -----------
    subject: str
        the subject name
    complete_key_list: list
        the list of all experiment names
    load_times: dict
        keys of this dictionary are the experiment names, values are tuples of (start-time, hours-to-load)
    paths: dict
        a dictionary of the paths to the TDT-blocks for each experiment
    channels: dict
        keys of this dictionary are the names of the TDT stores, values are the channels to load
    """
    return


a4 = {}
a4["subject"] = "ACR_4"
a4["complete_key_list"] = ["laser1-bl", "laser1-sd"]
a4["paths"] = acio.get_acr_paths(a4["subject"], a4["complete_key_list"])
a4["times"] = {}

a4["load_times"] = {}
a4["load_times"]["laser1-bl"] = (0, 18000)
a4["load_times"]["laser1-sd"] = (0, 0)
a4["channels"] = {}
a4["channels"]["EEGr"] = [1, 2]
a4["channels"]["EMGr"] = [1, 2]

a9 = {}
a9["subject"] = "ACR_9"
a9["complete_key_list"] = ["control1", "laser1"]
a9["paths"] = acio.get_acr_paths(a9["subject"], a9["complete_key_list"])
a9["times"] = {}

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

a9["times"]["control1"] = a9_times_c1
a9["times"]["laser1"] = a9_times_l1

a9["load_times"] = {}
a9["load_times"]["control1"] = (0, 0)
a9["load_times"]["laser1"] = (0, 0)

a9["channels"] = {}
a9["channels"]["EEGr"] = [1, 2]
a9["channels"]["NNXr"] = list(range(1, 17))
a9["channels"]["LFP_"] = list(range(1, 17))

a10 = {}
a10["subject"] = "ACR_10"
a10["complete_key_list"] = ["laser1", "laser1-bl"]
a10["paths"] = acio.get_acr_paths(a10["subject"], a10["complete_key_list"])
a10["start_times"] = {}
a10["start_times"]["laser1"] = 4158
a10["start_times"]["laser1-bl"] = 0
a10["load_times"] = {}
a10["load_times"]["laser1"] = (0, 0)
a10["load_times"]["laser1-bl"] = (0, 0)
a10["channels"] = {}
a10["channels"]["EEGr"] = [1, 2]

a11 = {}
a11["subject"] = "ACR_11"
a11["complete_key_list"] = ["laser1", "laser1-bl"]
a11["paths"] = acio.get_acr_paths(a11["subject"], a11["complete_key_list"])
a11["start_times"] = {}
a11["start_times"]["laser1"] = 5974
a11["start_times"]["laser1-bl"] = 0
a11["load_times"] = {}
a11["load_times"]["laser1"] = (0, 0)
a11["load_times"]["laser1-bl"] = (0, 0)
a11["channels"] = {}
a11["channels"]["EEGr"] = [1, 2]

a12 = {}
a12["subject"] = "ACR_12"
a12["complete_key_list"] = ["control1-bl", "laser1-bl", "control1", "laser1"]
a12["paths"] = acio.get_acr_paths(a12["subject"], a12["complete_key_list"])
a12["load_times"] = {}
a12["load_times"]["control1-bl"] = (0, 0)
a12["load_times"]["laser1-bl"] = (0, 0)
a12["load_times"]["control1"] = (0, 0)
a12["load_times"]["laser1"] = (0, 0)
a12["channels"] = {}
a12["channels"]["EEGr"] = [1, 2]
a12["channels"]["NNXr"] = list(range(1, 17))
a12["channels"]["LFP_"] = list(range(1, 17))

a13 = {}
a13["subject"] = "ACR_13"
a13["complete_key_list"] = ["laser1-bl", "laser1", "laser1-post1", "laser1-post2"]
a13["paths"] = acio.get_acr_paths(a13["subject"], a13["complete_key_list"])
a13["load_times"] = {}
a13["load_times"]["laser1-bl"] = (0, 23)
a13["load_times"]["laser1"] = (14027, 6)
a13["channels"] = {}
a13["channels"]["EEGr"] = [1, 2]
a13["channels"]["NNXr"] = list(range(1, 17))
a13["channels"]["LFP_"] = [3, 13]
a13["channels"]["LFPo"] = [3, 13]
a13["channels"]["NNXo"] = list(range(1, 17))

a13["times"] = {}
a13["times"]["laser1-pi"] = (16367.83562112, 18097.09928446)


a14 = {}
a14["subject"] = "ACR_14"
a14["complete_key_list"] = ["laser1-bl", "laser1", "laser1-post1", "laser1-post2"]
a14["load_times"] = {}
a14["load_times"]["laser1-bl"] = (0, 12)
a14["load_times"]["laser1"] = (13999, 6)
a13["load_times"]["laser1-post1"] = (0, 6)
a13["load_times"]["laser1-post2"] = (0, 6)

a14["paths"] = acio.get_acr_paths(a14["subject"], a14["complete_key_list"])
a14["channels"] = {}
a14["channels"]["EEGr"] = [1, 2]
a14["channels"]["NNXr"] = list(range(1, 17))
a14["channels"]["LFP_"] = [2, 14]
a14["channels"]["LFPo"] = [2, 14]
a14["channels"]["NNXo"] = list(range(1, 17))

a14["times"] = {}
a14["times"]["laser1-pi"] = (15764.02395704, 17515.97520008)
