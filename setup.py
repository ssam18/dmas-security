from setuptools import setup, find_packages

setup(
    name="dmas-security",
    version="0.1.0",
    description="Decentralized Multi-Agent Swarms for Autonomous Grid Security in Industrial IoT",
    author="Samaresh Kumar Singh, Joyjit Roy",
    author_email="ssam3003@gmail.com",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "torch>=2.0.0",
        "pyyaml>=6.0",
        "pyahocorasick>=2.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-asyncio>=0.21.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
