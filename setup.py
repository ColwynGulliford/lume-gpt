
import versioneer
from setuptools import setup, find_packages
from os import path, environ


cur_dir = path.abspath(path.dirname(__file__))

with open(path.join(cur_dir, 'requirements.txt'), 'r') as f:
    requirements = f.read().split()

setup(

    name='lume-gpt',
    version = versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(), 
    packages=find_packages(), 
    package_dir={'lume-gpt':'lume-gpt'},
    url='https://github.com/ColwynGulliford/lume-gpt',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=requirements,
    include_package_data=True,
    python_requires='>=3.9'
)
