from setuptools import find_packages, setup

setup(
    name="acr",
    version="0.1",
    description="Analysis tools for ACR optogenetic inhibition",
    url="http://github.com/kortdriessen/acr",
    author="Kort Driessen",
    author_email="driessen2@wisc.edu",
    license="MIT",
    # packages=find_packages("src"),
    # package_data={"acr": ["py.typed", "plot_styles/*.mplstyle"]},
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "streamlit",
        "plotly",
        "openpyxl",
        "jupyterlab",
        "ipykernel",
        "h5py",
        "python-benedict",
        "statsmodels",
        "XlsxWriter",
        "dask",
        "dask-image",
        "xhistogram",
        "pingouin",
    ],  # needs kdephys
    zip_safe=False,
)
