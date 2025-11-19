# GGL-MB-Score 

This repository provides an implementation for generating GGL-MB features to predict molecular binding affinity. The core innovation is the use of triple-based interaction scores that capture complex protein-ligand interactions more effectively than traditional pairwise methods.
The folder `src` contains the main source code. The code `get_features.py` can be used to generate features for a given protein-ligand dataset. 

## Package Requirement
- NumPy
- SciPy
- Pandas
- BioPandas
- RDKit


## Simple Example
Assume we want to generate the features for the PDBbind v2016 general set with exponential kernel type and parameters $p$ = 2.5, $\tau=1.5$ and $c$ = 6.0 Assume also the structures of the dataset are in the directory `../PDBbind_v2016_general_Set` and we wish to save the features in the directory `../features`.

GGL features are adopted from https://github.com/MathIntelligence/GGL-ETA-Score

```shell
python3 src/get_features.py \ 
    -p 2.5 \
    -tau 1.5 \
    -c 6.0 \ 
    -t "lorentz_kernel" \ 
    -f './csv_data_file/PDBbindv2016_GeneralSet.csv' \
    -dd './PDBbind_v2016_general_set' \
    -fd './features' \
    [-set ...]
```
