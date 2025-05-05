from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read long description
with open('README.md') as f:
    long_description = f.read()

setup(
    name="rag-csd",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Retrieval-Augmented Generation with Computational Storage Devices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rag-csd",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "rag-csd-create-db=rag_csd.scripts.create_vector_db:main",
            "rag-csd-evaluate=rag_csd.scripts.evaluate_retrieval:main",
        ],
    },
    include_package_data=True,
)