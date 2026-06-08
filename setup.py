from setuptools import setup, find_packages

setup(
    name="multi_agent_attribution",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "shap>=0.42.0",
        "pandas>=2.0.0",
    ],
    author="Your Name",
    description="Attribution methods for multi-agent interactive systems",
    license="MIT",
)
