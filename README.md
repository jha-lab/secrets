# SECRETS: Subject-Efficient Clinical Randomized Controlled Trials using Synthetic Intervention

SECRETS is a tool to increase the power of a clinical RCT by simulating a cross-over trial design using [Synthetic Interventions](https://arxiv.org/pdf/2006.07691.pdf). 


## Table of Contents
- [Environment Setup](#environment-setup)
- [Dataset request](#data-request)
- [Run SECRETS](#run-secrets)
- [Developer](#developer)
- [Cite this work](#cite-this-work)
- [License](#license)

## Environment setup

To setup the python environment with `pip`, see the `requirements.txt` file. 

## Dataset Request
To obtain the datasets used in the paper (see details in paper), submit requests to [NINDS] (https://www.ninds.nih.gov/current-research/research-funded-ninds/clinical-research/archived-clinical-research-datasets).

Download these into the datasets directory.

## Run SECRETS

From the secrets directory, run the following command. This will launch jobs using slurm to run SECRETS on the MGTX dataset. 
See comments within files to adapt script for experiments.

```shell
cd scripts
./gen_job_slurm_mgtx.sh
```

To run SECRETS on other datasets (e.g., CHAMP, ICARE), use the respective scripts or follow the template of gen_job_slurm_mgtx.* and syn_ctrl_mgtx.py to run on new datasets.


## Developer

[Sayeri Lala](https://github.com/slala2121). For any questions, comments or suggestions, please reach me at [slala@princeton.edu](mailto:slala@princeton.edu).

## Cite this work

Cite our work using the following bitex entry:
```bibtex
@article{lala2024secrets,
  title={SECRETS: Subject-efficient clinical randomized controlled trials using synthetic intervention},
  author={Lala, Sayeri and Jha, Niraj K.},
  journal={Contemporary Clinical Trials Communications},
  year={2024},
  publisher={Elsevier}
}
```

## License

Copyright (c) 2024, Sayeri Lala and Jha Lab.
All rights reserved.

See License file for more details.
