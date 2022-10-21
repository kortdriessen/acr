from setuptools import setup

setup(
    name="acr",
    version="0.1",
    description="Analysis tools for ACR optogenetic inhibition",
    url="http://github.com/kortdriessen/acr",
    author="Kort Driessen",
    author_email="driessen2@wisc.edu",
    license="MIT",
    packages=["acr"],
    install_requires=["streamlit", "plotly"],  # needs kdephys + ecephys
    zip_safe=False,
)
