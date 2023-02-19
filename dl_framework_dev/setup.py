import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dml-automl",
    version="0.1.5.0",
    author="shuangxi.fan",
    author_email="fanshuangxi@didiglobal.com",
    description="dml-automl",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.xiaojukeji.com/dml/dml-automl.git",
    project_urls={
        "Bug Tracker": "https://git.xiaojukeji.com/dml/dml-automl/issues",
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6"
)
