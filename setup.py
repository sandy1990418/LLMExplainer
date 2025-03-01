import setuptools
import os

# Version will be read from version.py
version = ""
name = "explainer"
# Fetch Version
with open(os.path.join(name, "__version__.py"), encoding="utf-8") as f:
    code = compile(f.read(), f.name, "exec")
    exec(code)

# Fetch ReadMe
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Use requirements.txt to set the install_requires
with open("requirements.txt", encoding="utf-8") as f:
    install_requires = [line.strip() for line in f]

setuptools.setup(
    name="LLMExplainer",  # noqa: F821
    version=version,  # noqa: F821
    author="Sandy Chen",
    author_email="sandy1990418@gmail.com",
    description="A tool for LLM explainability, "
    "focusing on attention mechanism and gradient "
    "analysis to enhance model transparency.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sandy1990418/LLMExplainer",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
)
