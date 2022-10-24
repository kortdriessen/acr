from asyncio import streams
import xarray as xr
import yaml
import acr.info_pipeline as aip
import tdt

subject = "ACR_14"
info = aip.load_subject_info(subject)

yaml_dir = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/times.yml"
root = f"/Volumes/opto_loc/Data/{subject}"

ty = yaml.load(open(yaml_dir, "r"), Loader=yaml.FullLoader)

# ----------------------sdpi-------------------
sorting = "sdpi"
recs = ["sdpi-bl", "sdpi", "sdpi-post"]

yaml_dir = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/times.yml"
root = f"/Volumes/opto_loc/Data/{subject}"

ty = yaml.load(open(yaml_dir, "r"), Loader=yaml.FullLoader)

ty[sorting] = {}


for rec in recs:
    ty[sorting][rec] = {}
    ty[sorting][rec]["info_start"] = info["rec_times"][rec]["start"]
    ty[sorting][rec]["info_end"] = info["rec_times"][rec]["end"]
    ty[sorting][rec]["info_duration"] = info["rec_times"][rec]["duration"]

    path = f"{root}/{subject}-{rec}/"
    data = tdt.read_block(path, store=["NNXr", "NNXo"], channel=[1], t1=0, t2=0)
    o_time = len(data.streams.NNXo.data) / data.streams.NNXo.fs
    p_time = len(data.streams.NNXr.data) / data.streams.NNXr.fs

    ty[sorting][rec]["optrode_total_samples"] = str(len(data.streams.NNXo.data))
    ty[sorting][rec]["probe_total_samples"] = str(len(data.streams.NNXr.data))

    ty[sorting][rec]["calc_opto_duration"] = str(o_time)
    ty[sorting][rec]["calc_probe_duration"] = str(p_time)

    xr_probe = xr.open_dataarray(f"{root}/{rec}-NNXo.nc").sel(channel=1)

    xr_opto = xr.open_dataarray(f"{root}/{rec}-NNXr.nc").sel(channel=1)
    ty[sorting][rec]["xarray_probe_duration"] = str(xr_probe.time.values.max())
    ty[sorting][rec]["xarray_opto_duration"] = str(xr_opto.time.values.max())


# ----------------------laser1-------------------
sorting = "laser1"
recs = ["laser1-bl", "laser1", "laser1-post1"]

yaml_dir = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/times.yml"
root = f"/Volumes/opto_loc/Data/{subject}"

ty = yaml.load(open(yaml_dir, "r"), Loader=yaml.FullLoader)

ty[sorting] = {}


for rec in recs:
    ty[sorting][rec] = {}
    ty[sorting][rec]["info_start"] = info["rec_times"][rec]["start"]
    ty[sorting][rec]["info_end"] = info["rec_times"][rec]["end"]
    ty[sorting][rec]["info_duration"] = info["rec_times"][rec]["duration"]

    path = f"{root}/{subject}-{rec}/"
    data = tdt.read_block(path, store=["NNXr", "NNXo"], channel=[1], t1=0, t2=0)
    o_time = len(data.streams.NNXo.data) / data.streams.NNXo.fs
    p_time = len(data.streams.NNXr.data) / data.streams.NNXr.fs

    ty[sorting][rec]["optrode_total_samples"] = str(len(data.streams.NNXo.data))
    ty[sorting][rec]["probe_total_samples"] = str(len(data.streams.NNXr.data))

    ty[sorting][rec]["calc_opto_duration"] = str(o_time)
    ty[sorting][rec]["calc_probe_duration"] = str(p_time)

    xr_probe = xr.open_dataarray(f"{root}/{rec}-NNXo.nc").sel(channel=1)

    xr_opto = xr.open_dataarray(f"{root}/{rec}-NNXr.nc").sel(channel=1)
    ty[sorting][rec]["xarray_probe_duration"] = str(xr_probe.time.values.max())
    ty[sorting][rec]["xarray_opto_duration"] = str(xr_opto.time.values.max())


# ----------------------swi-------------------
sorting = "swi"
recs = ["swi-bl", "swi-sd", "swi"]

yaml_dir = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/times.yml"
root = f"/Volumes/opto_loc/Data/{subject}"


ty[sorting] = {}


for rec in recs:
    ty[sorting][rec] = {}
    ty[sorting][rec]["info_start"] = info["rec_times"][rec]["start"]
    ty[sorting][rec]["info_end"] = info["rec_times"][rec]["end"]
    ty[sorting][rec]["info_duration"] = info["rec_times"][rec]["duration"]

    path = f"{root}/{subject}-{rec}/"
    data = tdt.read_block(path, store=["NNXr", "NNXo"], channel=[1], t1=0, t2=0)
    o_time = len(data.streams.NNXo.data) / data.streams.NNXo.fs
    p_time = len(data.streams.NNXr.data) / data.streams.NNXr.fs

    ty[sorting][rec]["optrode_total_samples"] = str(len(data.streams.NNXo.data))
    ty[sorting][rec]["probe_total_samples"] = str(len(data.streams.NNXr.data))

    ty[sorting][rec]["calc_opto_duration"] = str(o_time)
    ty[sorting][rec]["calc_probe_duration"] = str(p_time)

    xr_probe = xr.open_dataarray(f"{root}/{rec}-NNXo.nc").sel(channel=1)

    xr_opto = xr.open_dataarray(f"{root}/{rec}-NNXr.nc").sel(channel=1)
    ty[sorting][rec]["xarray_probe_duration"] = str(xr_probe.time.values.max())
    ty[sorting][rec]["xarray_opto_duration"] = str(xr_opto.time.values.max())

# write the yaml file
yaml.dump(ty, open(yaml_dir, "w"))
