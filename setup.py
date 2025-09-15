from setuptools import setup, find_packages

setup(
    name='falafel',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'flask',
        'gunicorn',
        # Add other dependencies as needed
    ],
    entry_points={
        'console_scripts': [
            'start_myapp=start_gunicorn:main',
        ],
    },
)
