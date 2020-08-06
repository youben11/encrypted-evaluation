import setuptools
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(
    name="eeval",
    version="0.1.0",
    author="Ayoub Benaissa",
    author_email="ayouben9@gmail.com",
    install_requires=read("requirements.txt").split("\n"),
    description="Client/Server framework for encrypted machine learning evaluation",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    keywords="homomorphic encryption machine learning",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(include=["eeval", "eeval.*"]),
    url="https://github.com/youben11/encrypted-evaluation",
    # tests_require=["pytest"],
    entry_points={"console_scripts": ["eeval = eeval.__main__:run_cli",]},
    license="GPL3",
)
