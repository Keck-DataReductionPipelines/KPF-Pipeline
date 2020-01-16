from setuptools import setup, find_packages
import re

def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
                       open(project + '/__init__.py').read())
    return result.group(1)

# reqs = []
# for line in open('requirements.txt', 'r').readlines():
#     reqs.append(line)

setup(
    name="kpfpipe",
    version=get_property('__version__', 'kpfpipe'),
    author="BJ Fulton, Arpita Roy, Andrew Howard",
    packages=find_packages(),
    entry_points={'console_scripts': ['kpf=kpfpipe.tfacli:main']},
    # install_requires=reqs,
    # data_files=[
    #     (
    #         'rvsearch_example_data',
    #         [
    #             'example_data/HD128311.csv',
    #             'example_data/recoveries.csv'
    #         ]
    #     )
    # ],
)
