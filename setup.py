import setuptools
from setuptools import find_packages, setup
from typing import List

CONNECTOR= "-e ."
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements= file_obj.readlines()
        requirements=[i.replace("\n","") for i in requirements]
        
        if CONNECTOR in requirements:
            requirements.remove(CONNECTOR)
    
    return requirements

with open("README.md", 'r', encoding="utf-8") as f:
    long_description= f.read()

setup(
    name="SupplyChain_MLWorkflow Project",
    version="0.0.01",
    author="Subinoy Bera",
    author_email="subinoyberadgp@gmail.com",
    description="A small end to end python package for machine learning app",
    long_description=long_description,
    long_description_content="text/markdown",
    url="https://github.com/SubinoyBera/SupplyChain_MLWflow",
    project_urls={
        "Bug Tracker": "https://github.com/SubinoyBera/SupplyChain_MLWflow/issues"
    },
    install_requires=get_requirements("requirements.txt"),
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    extras_require={'dev':['jupyter']},
    python_requires=">=3.11.0",
    license="Apache 2.0"
)