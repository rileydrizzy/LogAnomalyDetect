# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
if __name__ == "__main__":
    setup(
        name="kaggle-sandbox",
        version=_about.__version__,
        description=_about.__docs__,
        author=_about.__author__,
        author_email=_about.__author_email__,
        url=_about.__homepage__,
        license=_about.__license__,
        packages=find_namespace_packages(where="src"),
        package_dir={"": "src"},
        long_description=_load_long_description(
            _about.__homepage__, _about.__version__
        ),
        long_description_content_type="text/markdown",
        include_package_data=True,
        zip_safe=False,
        keywords=["deep learning", "pytorch", "AI"],
        python_requires=">=3.6",
        setup_requires=[],
        install_requires=_load_requirements(_PATH_ROOT),
        project_urls={
            "Source Code": _about.__homepage__,
        },
