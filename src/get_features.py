import numpy as np
from tqdm import tqdm
import pandas as pd
from itertools import product
import argparse
import time
import sys
from get_scores import *

atoms_pro = ['C', 'N', 'O', 'S']
atoms_lig = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']

class GET_PAIRWISE_FEATURES:
    def __init__(self, args):
        self.cutoff = args.cutoff
        self.path_to_csv = args.path_to_csv
        self.data_folder = args.data_folder
        self.feature_folder = args.feature_folder
        self.type = args.type
        self.tau = args.tau
        self.power = args.power

    def get_pairwise_features(self, parameters):
        df_pdbids = pd.read_csv(self.path_to_csv)
        pdbids = df_pdbids['PDBID'].tolist()
        pks = df_pdbids['pK'].tolist()

        Kernel = KernelFunction(kernel_type=parameters['type'],
                                power=parameters['power'], tau=parameters['tau'])

        PW = PAIRWISE_SCORE(Kernel=Kernel, cutoff=parameters['cutoff'])
        for index, _pdbid in enumerate(pdbids):
            lig_file = f'{self.data_folder}/{_pdbid}/{_pdbid}_ligand.mol2'
            pro_file = f'{self.data_folder}/{_pdbid}/{_pdbid}_protein.pdb'

            pw_score = PW.get_pairwise_score(pro_file, lig_file)

            atom_pairs = pw_score['ATOM_PAIR'].tolist()
            features = pw_score.columns[1:].tolist()

            pairwise_features = [i[0]+'_'+i[1]
                                 for i in product(atom_pairs, features)]
            feature_values = pw_score.drop(
                ['ATOM_PAIR'], axis=1).values.flatten()
            if index == 0:
                df_features = pd.DataFrame(columns=pairwise_features)
            df_features.loc[index] = feature_values

        df_features.insert(0, 'PDBID', pdbids)
        df_features.insert(1, 'pK', pks)

        return df_features


class GET_TRIPLET_FEATURES:
    def __init__(self, args):
        self.cutoff = args.cutoff
        self.path_to_csv = args.path_to_csv
        self.data_folder = args.data_folder
        self.feature_folder = args.feature_folder
        self.power = args.power

    def get_trip_features(self, parameters):
        df_pdbids = pd.read_csv(self.path_to_csv)
        pdbids = df_pdbids['PDBID'].tolist()
        pks = df_pdbids['pK'].tolist()

        MRI = TRIPLET_SCORE(self.cutoff, False, parameters['tau'], parameters['type'])
        for index, _pdbid in enumerate(tqdm(pdbids)):
            lig_file = f'{self.data_folder}/{_pdbid}/{_pdbid}_ligand.mol2'
            pro_file = f'{self.data_folder}/{_pdbid}/{_pdbid}_protein.pdb'
            tri_score = MRI.get_triplet(pro_file, lig_file)
            atom_pairs = tri_score['type'].tolist()
            print(len(atom_pairs))
            features = tri_score.columns[1:].tolist()
            print(len(features))
            pairwise_features = [f"{pair}_{feature}" for pair, feature in product(atom_pairs, features) if pair != 'NA']
            print("this is len",len(pairwise_features))
            feature_values = tri_score.drop(['type'], axis=1).values.flatten()
            if index == 0:
                df_features = pd.DataFrame(columns=pairwise_features)
            df_features.loc[index] = feature_values

        df_features.insert(0, 'PDBID', pdbids)
        df_features.insert(1, 'pK', pks)

        return df_features 

class GET_GGL_FEATURES:

    def __init__(self, args):
        self.cutoff = args.cutoff
        self.path_to_csv = args.path_to_csv
        self.data_folder = args.data_folder
        self.feature_folder = args.feature_folder


    def get_ggl_features(self, parameters):

        df_pdbids = pd.read_csv(self.path_to_csv)
        pdbids = df_pdbids['PDBID'].tolist()
        pks = df_pdbids['pK'].tolist()

        Kernel = KernelFunction(kernel_type=parameters['type'],
                                power=parameters['power'], tau=parameters['tau'])

        GGL = SYBYL_GGL(Kernel=Kernel, cutoff=parameters['cutoff'])
        for index, _pdbid in enumerate(pdbids):
            lig_file = f'{self.data_folder}/{_pdbid}/{_pdbid}_ligand.mol2'
            pro_file = f'{self.data_folder}/{_pdbid}/{_pdbid}_protein.pdb'

            ggl_score = GGL.get_ggl_score(pro_file, lig_file)

            atom_pairs = ggl_score['ATOM_PAIR'].tolist()
            features = ggl_score.columns[1:].tolist()

            pairwise_features = [i[0]+'_'+i[1]
                                 for i in product(atom_pairs, features)]
            feature_values = ggl_score.drop(
                ['ATOM_PAIR'], axis=1).values.flatten()
            if index == 0:
                df_features = pd.DataFrame(columns=[pairwise_features])
            df_features.loc[index] = feature_values

        df_features.insert(0, 'PDBID', pdbids)
        df_features.insert(1, 'pK', pks)

        return df_features   

def merge_features(pair_df, triplet_df, ggl_df):
    merged_df = pair_df.merge(triplet_df, on='PDBID', how='inner')
    return merged_df


def get_args(args):

    parser = argparse.ArgumentParser(description="Get GGL-MB Features")

    parser.add_argument('-c', '--cutoff', help='distance cutoff to define binding site',
                        type=float, default=12.0)
    parser.add_argument('-t', '--type', help='kernel type',
                        type=str, default='lorentz_kernel')
    parser.add_argument('-tau', '--tau', help='tau',
                        type=float, default=0.5)
    parser.add_argument('-p', '--power', help='power',
                        type=float, default=2.5)
    parser.add_argument('-f', '--path_to_csv',
                        help='path to CSV file containing PDBIDs and pK values')
    parser.add_argument('-dd', '--data_folder', type=str,
    					help='path to data folder directory')
    parser.add_argument('-fd', '--feature_folder', type=str,
    					help='path to the directory where features will be saved')

    
    args = parser.parse_args()

    return args


def cli_main():
    args = get_args(sys.argv[1:])

    parameters = {
        'type': args.type,
        'power': args.power,
        'tau': args.tau,
        'cutoff': args.cutoff,
        'multi': args.parallel_running
    }

    GGL_Features = GET_GGL_FEATURES(args)
    ggl_df = GGL_Features.get_ggl_features(parameters)

    TRI_FEATURES = GET_TRIPLET_FEATURES(args)
    tri_df = TRI_FEATURES.get_trip_features(parameters)

    PAIR_FEATURES = GET_PAIRWISE_FEATURES(args)
    pair_df = PAIR_FEATURES.get_pairwise_features(parameters)
    merged_df = merge_features(pair_df, tri_df, ggl_df)

    merged_df.to_csv(f"{args.feature_folder}/mb_features.csv", index=False, float_format='%.5f')
    ggl_df.to_csv(f"{args.feature_folder}/ggl_features.csv", index=False, float_format='%.5f')



if __name__ == "__main__":

    t0 = time.time()

    cli_main()

    print('Done!')
    print('Elapsed time: ', time.time()-t0)
