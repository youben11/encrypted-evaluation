import setuptools


setuptools.setup(
    name="eeval",
    version="0.1.0",
    author="Ayoub Benaissa",
    author_email="ayouben9@gmail.com",
    description="Client/Server for encrypted machine learning evaluation",
    keywords="homomorphic encryption machine learning",
    packages=setuptools.find_packages(
        include=["eeval", "eeval.*"]
    ),
    url="https://github.com/youben11/encrypted_evaluation",
    # tests_require=["pytest"],
    entry_points={
        'console_scripts': [
            'eeval = eeval.__main__:run_cli',
        ]
    }
)
