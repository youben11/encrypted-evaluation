import setuptools


setuptools.setup(
    name="encrypted-evaluation",
    version="0.1.0",
    author="Ayoub Benaissa",
    author_email="ayouben9@gmail.com",
    description="Client/Server for encrypted machine learning evaluation",
    keywords="homomorphic encryption machine learning",
    packages=setuptools.find_packages(
        include=["encrypted_evaluation", "encrypted_evaluation.*"]
    ),
    url="https://github.com/youben11/encrypted_evaluation",
    # tests_require=["pytest"],
)
