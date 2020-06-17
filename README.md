# Joint Policy-Value Learning for Recommendation
Source code for our paper "Joint Policy-Value Learning for Recommendation" published at KDD 2020.

## Reproducibility
To generate a virtual Python environment tha holds all the packages our work relies on, run:

    virtualenv -p python3 dual_bandit
    source dual_bandit/bin/activate
    pip3 install -r requirements.txt
    
Now, you can run any of the three ''RunABTest'' files to run all experimental results corresponding to a column in Fig. 2 of the paper.
For example:

    python3 src/RunABTest_Uniform.py


## Paper
If you use our code in your research, please remember to cite our paper:

    @inproceedings{JeunenKDD2020,
      author = {Jeunen, Olivier and Rohde, David and Vasile, Flavian and Bompaire, Martin},
      title = {Joint Policy-Value Learning for Recommendation},
      booktitle = {Proceedings of the 26th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
      series = {KDD '20},
      year = {2020},
      publisher = {ACM},
    }
