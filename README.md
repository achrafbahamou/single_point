# single_point

This is the code to for the paper [Optimal Pricing with a Single Point][opt-paper].

[opt-paper]: https://arxiv.org/abs/2103.05611

## Specification of dependencies

Python 3.7.6; 

cvxopt==1.3.0, matplotlib==3.4.3, numpy==1.20.3, scipy==1.7.1, seaborn==0.11.2.


## How to get results

1- Install dependencies from requirements.txt

2- Run each cell in reproduce_results.ipynb to reproduce the corresponding result described in the heading comment of the cell.

3- The figure outputs are stored in the folder figures the data output is stored in dict_results.pickle


The package single_point contains the code used in producing the results in the notebook reproduce_results.ipynb. 

## BibTeX

```
@article{DBLP:journals/corr/abs-2103-05611,
  author    = {Amine Allouah and
               Achraf Bahamou and
               Omar Besbes},
  title     = {Optimal Pricing with a Single Point},
  journal   = {CoRR},
  volume    = {abs/2103.05611},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.05611},
  eprinttype = {arXiv},
  eprint    = {2103.05611},
  timestamp = {Tue, 16 Mar 2021 11:26:59 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2103-05611.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
