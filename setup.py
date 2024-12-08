import setuptools

with open("requirements.txt") as fh:
    requirements = fh.read().splitlines()

setuptools.setup(
    name="autodiff",
    version="0.0.1",
    description="An autodiff package",
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    zip_safe=True,
)