"""doc
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.1.0"

PROJECT_REPO_NAME = "LogAnomalyDetect"
AUTHOR_NAME = "Ladipo Ipadeola"
AUTHOR_USER_NAME = "rileydrizzy"
AUTHOR_EMAIL = "ipadeolaoladipo@outlook.com"

if __name__ == "__main__":
    setup(
        name=PROJECT_REPO_NAME,
        version=__version__,
        license="MIT",
        author=AUTHOR_NAME,
        author_email=AUTHOR_EMAIL,
        description="A machine learning model to predict log of softwares",
        long_description=long_description,
        long_description_content="text/markdown",
        url=f"https://github.com/{AUTHOR_USER_NAME}/{PROJECT_REPO_NAME}",
        project_urls={
            "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{PROJECT_REPO_NAME}/issues",
        },
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        python_requires=">=3.10",
        keywords=["deep learning", "tensorflow", "AI"],
    )
