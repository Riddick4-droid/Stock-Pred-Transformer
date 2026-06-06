from pathlib import Path
from setuptools import setup, find_packages

def read_requirements(path: str = "requirements.txt") -> list[str]:
    requirements = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)
    return requirements

setup(
    name="stock-pred-transformer",
    version="0.1.0",
    description="Stock price prediction with transformer architecture",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Riddick Mensah",
    author_email="riddick.mensah@yahoo.com",
    url="https://github.com/riddick4-droid/Stock-Pred-Transformer",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=read_requirements("requirements.txt"),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)