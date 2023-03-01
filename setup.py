from setuptools import setup

setup(

    name='anatomy-lookup',
    version='0.0.3',
    description='Anatomy lookup for UBERON and ILX',
    url='http://github.com/napakalas/anatomy-lookup',
    author='Yuda Munarko',
    author_email='yuda.munarko@gmail.com',
    license='GPL v.3',
    packages=['anatomy_lookup'],
    zip_safe=False,
    install_requires=[
        'torch>=1.13.0',
        'tqdm',
        'sentence-transformers>=2.2.2',
        'rdflib>=6.0.0',
        'xlsxwriter>=3.0.1',
    ],
    # extras_require={'indexer': ['rdflib>=6.0.0',],},
      
)
#===============================================================================