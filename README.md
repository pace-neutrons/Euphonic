# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/pace-neutrons/Euphonic/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                        |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|-------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| euphonic/\_\_init\_\_.py                    |       10 |        0 |        0 |        0 |    100% |           |
| euphonic/brille.py                          |      100 |        5 |       28 |        7 |     91% |214-215, 219-\>222, 238, 249-\>258, 251, 259-\>267, 262 |
| euphonic/broadening.py                      |      100 |        5 |       32 |        3 |     91% |94-\>97, 164, 242-243, 305-306 |
| euphonic/cli/\_\_init\_\_.py                |        0 |        0 |        0 |        0 |    100% |           |
| euphonic/cli/brille\_convergence.py         |      132 |        4 |       28 |        3 |     96% |115, 127-128, 177 |
| euphonic/cli/dispersion.py                  |       46 |        0 |       12 |        0 |    100% |           |
| euphonic/cli/dos.py                         |       64 |        2 |       24 |        2 |     95% |   80, 108 |
| euphonic/cli/intensity\_map.py              |       59 |        0 |       22 |        2 |     98% |87-\>90, 102-\>105 |
| euphonic/cli/optimise\_dipole\_parameter.py |       52 |        0 |       14 |        0 |    100% |           |
| euphonic/cli/powder\_map.py                 |      169 |       12 |       44 |        3 |     93% |129, 186-188, 264-265, 321-322, 343-344, 347-348 |
| euphonic/cli/show\_sampling.py              |       54 |        1 |       24 |        0 |     99% |        20 |
| euphonic/cli/utils/\_\_init\_\_.py          |        9 |        0 |        0 |        0 |    100% |           |
| euphonic/cli/utils/\_band\_structure.py     |       55 |        0 |       12 |        0 |    100% |           |
| euphonic/cli/utils/\_cli\_parser.py         |      126 |        0 |       56 |        0 |    100% |           |
| euphonic/cli/utils/\_dw.py                  |        9 |        0 |        0 |        0 |    100% |           |
| euphonic/cli/utils/\_grids.py               |       26 |        0 |        8 |        0 |    100% |           |
| euphonic/cli/utils/\_kwargs.py              |       10 |        0 |        2 |        0 |    100% |           |
| euphonic/cli/utils/\_loaders.py             |       78 |        2 |       38 |        2 |     97% |60-\>65, 112-113 |
| euphonic/cli/utils/\_pdos.py                |       23 |        0 |       10 |        0 |    100% |           |
| euphonic/cli/utils/\_plotting.py            |       27 |        0 |       12 |        0 |    100% |           |
| euphonic/crystal.py                         |      115 |        7 |       20 |        3 |     90% |105-\>107, 107-\>exit, 290-306 |
| euphonic/debye\_waller.py                   |       46 |        0 |        0 |        0 |    100% |           |
| euphonic/force\_constants.py                |      593 |        0 |      168 |        3 |     99% |47-\>49, 1070-\>1107, 1116-\>1142 |
| euphonic/io.py                              |       79 |        0 |       46 |        3 |     98% |15-\>17, 15-\>exit, 18-\>exit |
| euphonic/isotopes/\_\_init\_\_.py           |        6 |        0 |        0 |        0 |    100% |           |
| euphonic/isotopes/\_core.py                 |      120 |       21 |       26 |        0 |     77% |231-238, 249-276, 282-294 |
| euphonic/isotopes/\_csv.py                  |      162 |        4 |       50 |        5 |     96% |22-\>25, 25-\>28, 28-\>31, 124-\>exit, 206-207, 395-396 |
| euphonic/isotopes/\_legacy.py               |       10 |        0 |        2 |        0 |    100% |           |
| euphonic/isotopes/data/\_\_init\_\_.py      |        0 |        0 |        0 |        0 |    100% |           |
| euphonic/plot.py                            |      132 |        0 |       46 |        0 |    100% |           |
| euphonic/powder.py                          |       65 |        1 |       24 |        1 |     98% |       239 |
| euphonic/qpoint\_frequencies.py             |      126 |        0 |       22 |        1 |     99% | 239-\>258 |
| euphonic/qpoint\_phonon\_modes.py           |      200 |        4 |       56 |        4 |     97% |752, 791, 794, 797 |
| euphonic/readers/\_\_init\_\_.py            |        0 |        0 |        0 |        0 |    100% |           |
| euphonic/readers/castep.py                  |      289 |        7 |      108 |        5 |     97% |260-\>240, 412, 433-444, 688-693, 716-721 |
| euphonic/readers/phonopy.py                 |      296 |        3 |       72 |        0 |     99% |29, 391-392 |
| euphonic/readers/vasp.py                    |      199 |        0 |       44 |        0 |    100% |           |
| euphonic/sampling.py                        |       90 |        0 |       40 |        0 |    100% |           |
| euphonic/spectra/\_\_init\_\_.py            |        3 |        0 |        0 |        0 |    100% |           |
| euphonic/spectra/base.py                    |      422 |        6 |      118 |        7 |     98% |54-\>57, 136, 366-\>380, 558-562, 848-849, 1095, 1372-\>1377 |
| euphonic/spectra/collections.py             |      369 |       13 |       90 |        8 |     95% |101, 107, 110, 140-\>142, 140-\>exit, 198-\>207, 198-\>exit, 705-711, 997-998, 1070-1076, 1096-1102, 1105-1111 |
| euphonic/structure\_factor.py               |      115 |        3 |       16 |        1 |     97% |363, 416, 421 |
| euphonic/styles/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100% |           |
| euphonic/ureg/\_\_init\_\_.py               |        8 |        0 |        0 |        0 |    100% |           |
| euphonic/ureg/data/\_\_init\_\_.py          |        0 |        0 |        0 |        0 |    100% |           |
| euphonic/util.py                            |      220 |       11 |       54 |        6 |     94% |22-23, 263-267, 390-395, 521, 584-586, 724-\>739, 737-\>724, 740-744 |
| euphonic/validate.py                        |       51 |        1 |       20 |        1 |     97% |       201 |
| euphonic/version.py                         |        1 |        0 |        0 |        0 |    100% |           |
| euphonic/writers/\_\_init\_\_.py            |        0 |        0 |        0 |        0 |    100% |           |
| euphonic/writers/phonon\_website.py         |       61 |        2 |       10 |        1 |     96% |96-97, 201-\>205 |
| **TOTAL**                                   | **4930** |  **114** | **1398** |   **71** | **97%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/pace-neutrons/Euphonic/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/pace-neutrons/Euphonic/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pace-neutrons/Euphonic/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/pace-neutrons/Euphonic/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fpace-neutrons%2FEuphonic%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/pace-neutrons/Euphonic/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.