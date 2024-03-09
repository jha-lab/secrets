# SECRETS: Subject-Efficient Clinical Randomized Controlled Trials using Synthetic Intervention

<!-- comment 
![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FJHA-Lab%2Facceltran&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)
-->

SECRETS is a tool to increase the power of a clinical RCT by simulating a cross-over trial design using [Synthetic Intervention](https://arxiv.org/pdf/2006.07691.pdf). 


## Table of Contents
- [Environment Setup](#environment-setup)
- [Dataset request](#data-request)
- [Run SECRETS](#run-secrets)
- [Developer](#developer)
- [Cite this work](#cite-this-work)
- [License](#license)

## Environment setup

### Clone this repository and initialize sub-modules

```shell
git clone https://github.com/JHA-Lab/acceltran.git
cd ./acceltran/
git submodule init
git submodule update
```

### Setup python environment  

The python environment setup is based on conda. The script below creates a new environment named `txf_design-space`:
```shell
source env_setup.sh
```
For `pip` installation, we are creating a `requirements.txt` file. Stay tuned!


## Dataset Request
To obtain the datasets used in the paper (see details in paper), submit requests to [NINDS] (https://www.ninds.nih.gov/current-research/research-funded-ninds/clinical-research/archived-clinical-research-datasets).

## Run SECRETS



## Developer

[Sayeri Lala](https://github.com/slala21021). For any questions, comments or suggestions, please reach me at [slala@princeton.edu](mailto:slala@princeton.edu).

## Cite this work

Cite our work using the following bitex entry:
```bibtex
@article{lala2024secrets,
  title={SECRETS: Subject-efficient clinical randomized controlled trials using synthetic intervention},
  author={Lala, Sayeri and Jha, Niraj K},
  journal={Contemporary Clinical Trials Communications},
  pages={101265},
  year={2024},
  publisher={Elsevier}
}
```

## License

BSD-3-Clause. 
Copyright (c) 2024, Sayeri Lala and Jha Lab.
All rights reserved.

See License file for more details.
