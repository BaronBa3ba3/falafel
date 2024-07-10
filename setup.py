import io
import os
from setuptools import setup, find_packages


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("project_name", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]

setup(
    name='falafel',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        'console_scripts': [
            'start_myapp=start_gunicorn:main',
        ],
    },
)

### Avoid runnning : python setup.py install