import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from itertools import product, combinations
from biopandas.pdb import PandasPdb
from biopandas.mol2 import PandasMol2
from scipy.spatial import  KDTree
from collections import defaultdict
atoms_lig = ['C','N','O','S','P','F','Cl','Br','I']
atoms_pro = ['C','N','O','S']
radius_map = {
                            'N': 1.55,
                            'O': 1.5,
                            'F': 1.5,
                            'Si': 2.1,
                            'P': 1.85,
                            'S': 1.8,
                            'Cl': 1.7,
                            'C': 1.7,
                            'Br': 1.5,
                            'I': 1.5
                        }
class KernelFunction:
    def __init__(self, kernel_type, power, tau):
        self.kernel_type = kernel_type
        self.power = power
        self.tau = tau
        self.kernel_function = self.build_kernel_function(kernel_type)
    def build_kernel_function(self, kernel_type):
        if kernel_type[0].lower() == 'e':
            return self.exponential_kernel
        elif kernel_type[0].lower() == 'l':
            return self.lorentz_kernel
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
    def exponential_kernel(self, d, sum_radii):
        eta = self.tau*sum_radii
        return np.exp(-((d / eta) ** self.power))
    def lorentz_kernel(self, d, sum_radii):
        eta = self.tau*sum_radii
        return (eta / (eta + d)) ** self.power
    
class CLUSTER:
    def get_cluster(self): 
        cluster = []
        for pro1 in atoms_pro:
            for pro2 in atoms_pro:
                for lig in atoms_lig:
                        cluster.append({'pro1' :pro1, 'pro2': pro2,'lig' : lig})
                for pro3 in atoms_pro:
                    cluster.append({'pro1' :pro1, 'pro2': pro2,'pro3' : pro3})
            for lig1 in atoms_lig:
                for lig2 in atoms_lig:
                    cluster.append({'pro1' :pro1, 'lig1': lig1,'lig2' : lig2})
        for lig1 in atoms_lig:
            for lig2 in atoms_lig:
                for lig3 in atoms_lig:
                    cluster.append({'lig1' :lig1, 'lig2': lig2,'lig3' : lig3})
        return cluster
    @staticmethod
    def format_clustername(cluster):
        if 'pro1' in cluster and 'pro2' in cluster and 'lig' in cluster:
            formatted_name = f"{cluster['pro1']}(pro)-{cluster['pro2']}(pro)-{cluster['lig']}(lig)"
        elif 'pro1' in cluster and 'lig1' in cluster and 'lig2' in cluster:
            formatted_name = f"{cluster['pro1']}(pro)-{cluster['lig1']}(lig)-{cluster['lig2']}(lig)"
        elif 'pro1' in cluster and 'pro2' in cluster and 'pro3' in cluster:
            formatted_name = f"{cluster['pro1']}(pro)-{cluster['pro2']}(pro)-{cluster['pro3']}(pro)"
        elif 'lig1' in cluster and 'lig2' in cluster and 'lig3' in cluster:
            formatted_name = f"{cluster['lig1']}(lig)-{cluster['lig2']}(lig)-{cluster['lig3']}(lig)"
        else:
            formatted_name = str(cluster)  
        return formatted_name
    
class PARAMS:
    def __init__(self, cutoff,tau, kernel_type, power):
        self.cutoff = cutoff
        self.tau = tau
        self.kernel_type = kernel_type
        self.power = power

class TRIPLET_SCORE(PARAMS):
    def __init__(self, cutoff, tau, kernel_type, power):
        super().__init__(cutoff, tau, kernel_type, power)
        self.Kernel = KernelFunction(self.kernel_type, self.power, self.tau)
        self.cutoff = cutoff
        self.atom_lig = atoms_lig
        self.atom_pro = atoms_pro
        self.power = power
        self.cluster = CLUSTER()
        self.tau = tau
        self.kernel_type = kernel_type

    def pdb_to_df(self, pdb_file):
        ppdb = PandasPdb()
        ppdb.read_pdb(pdb_file)
        ppdb_all_df = ppdb.df['ATOM']
        ppdb_df = ppdb_all_df[ppdb_all_df['element_symbol'].apply(lambda x: x).isin(self.atom_pro)]
        atom_index = ppdb_df['atom_number']
        atom_element = ppdb_df['element_symbol']
        x, y, z = ppdb_df['x_coord'], ppdb_df['y_coord'], ppdb_df['z_coord']
        df = pd.DataFrame.from_dict({'ATOM_INDEX': atom_index, 
                                     'ATOM_ELEMENT': atom_element,
                                     'X': x, 
                                     'Y': y, 
                                     'Z': z,
                                     'radi': [radius_map.get(elem, np.nan) for elem in atom_element]})

        return df 

    def mol2_to_df(self, mol2_file):
        df_mol2 = PandasMol2().read_mol2(mol2_file).df
        filtered_df_mol2 = df_mol2.loc[df_mol2['atom_type'].apply(lambda x: x.split('.')[0]).isin(self.atom_lig)]
        df = pd.DataFrame({
            'ATOM_INDEX': filtered_df_mol2['atom_id'],
            'ATOM_ELEMENT': filtered_df_mol2['atom_type'].apply(lambda x: x.split('.')[0]),
            'X': filtered_df_mol2['x'],
            'Y': filtered_df_mol2['y'],
            'Z': filtered_df_mol2['z'],
            'radi':  filtered_df_mol2['atom_type'].apply(lambda x: radius_map.get(x.split('.')[0], np.nan))
        })

        return(df)

    def calculate_ri(self, matrix1, matrix2):
        if not matrix1 or not matrix2:
            return 0.0

        coords1 = np.array([atom[2:5] for atom in matrix1], dtype=np.float32)
        coords2 = np.array([atom[2:5] for atom in matrix2], dtype=np.float32)
        radi1 = np.array([atom[-1] for atom in matrix1], dtype=np.float32)
        radi2 = np.array([atom[-1] for atom in matrix2], dtype=np.float32)

        tree = KDTree(coords2)
        neighbors_list = tree.query_ball_point(coords1, self.cutoff)

        scores = []
        kernel = self.Kernel.kernel_function
        for i, neighbor_idxs in enumerate(neighbors_list):
            if not neighbor_idxs:
                continue

            idx1 = np.full(len(neighbor_idxs), i, dtype=int)
            idx2 = np.array(neighbor_idxs, dtype=int)

            d = np.linalg.norm(coords1[idx1] - coords2[idx2], axis=1)
            sum_radi =(radi1[idx1] + radi2[idx2])

            mask = d <= self.cutoff
            if np.any(mask):
                scores.append(np.sum(kernel(d[mask], sum_radi[mask])))

        return float(np.sum(scores))

    def calculate_ri_triplet(self, atom1s, atom2s, atom3s):
        if any([len(atom1s) == 0, len(atom2s) == 0, len(atom3s) == 0]):
            return {
                "SUM": 0.0,
                "MEAN": 0.0,
                "STD": 0.0,
                "MIN": 0.0,
                "MAX": 0.0
            }

        score1 = self.calculate_ri(atom1s, atom2s)
        score2 = self.calculate_ri(atom2s, atom3s)
        score3 = self.calculate_ri(atom1s, atom3s)

        triplet_scores = score1 + score2 + score3
        min_value = min(score1, score2, score3)
        max_value = max(score1, score2, score3)
        mean_value = np.mean([score1, score2, score3])  # Mean of all elements
        std_value = np.std([score1, score2, score3])
        score = {
            "SUM": triplet_scores,
            "MEAN": mean_value,
            "STD": std_value,
            "MIN": min_value,
            "MAX": max_value
        }
        return score

    def calculate_all_clusters(self, pro_df, lig_df, cluster_list):
        results = {}
        for clu in cluster_list:
            if 'pro1' in clu and 'pro2' in clu and 'lig' in clu:
                atom1 = pro_df[pro_df['ATOM_ELEMENT'] == clu['pro1']].values.tolist()
                atom2 = pro_df[pro_df['ATOM_ELEMENT'] == clu['pro2']].values.tolist()
                atom3 = lig_df[lig_df['ATOM_ELEMENT'] == clu['lig']].values.tolist()
            elif 'pro1' in clu and 'lig1' in clu and 'lig2' in clu:
                atom1 = pro_df[pro_df['ATOM_ELEMENT'] == clu['pro1']].values.tolist()
                atom2 = lig_df[lig_df['ATOM_ELEMENT'] == clu['lig1']].values.tolist()
                atom3 = lig_df[lig_df['ATOM_ELEMENT'] == clu['lig2']].values.tolist()

            elif 'pro1' in clu and 'pro2' in clu and 'pro3' in clu:
                atom1 = pro_df[pro_df['ATOM_ELEMENT'] == clu['pro1']].values.tolist()
                atom2 = pro_df[pro_df['ATOM_ELEMENT'] == clu['pro2']].values.tolist()
                atom3 = pro_df[pro_df['ATOM_ELEMENT'] == clu['pro3']].values.tolist()

            elif 'lig1' in clu and 'lig2' in clu and 'lig3' in clu:
                atom1 = lig_df[lig_df['ATOM_ELEMENT'] == clu['lig1']].values.tolist()
                atom2 = lig_df[lig_df['ATOM_ELEMENT'] == clu['lig2']].values.tolist()
                atom3 = lig_df[lig_df['ATOM_ELEMENT'] == clu['lig3']].values.tolist()
            scores = self.calculate_ri_triplet(atom1, atom2, atom3)
            results[CLUSTER().format_clustername(clu)] = scores
        return results

    def get_triplet(self, protein_file, ligand_file):
        protein_df = self.pdb_to_df(protein_file)
        ligand_df = self.mol2_to_df(ligand_file)
        triplet_scores = self.calculate_all_clusters(protein_df, ligand_df, self.cluster.get_cluster()) 
        return self._create_final_dataframe(triplet_scores)

    def _create_final_dataframe(self, triplet_scores):
        results = []
        for key, scores in triplet_scores.items():
            stats = {
                'SUM': scores['SUM'],
                'MIN': scores['MIN'], 
                'MAX': scores['MAX'],
                'MEAN': scores['MEAN'],
                'STD': scores['STD'],
            }
            results.append({'type': key, **stats})
        return pd.DataFrame(results)
    
class PAIRWISE_SCORE(PARAMS):
    protein_ligand_atom_types = [
        i[0]+"-"+i[1] for i in product(atoms_lig, atoms_pro)]
    def __init__(self, cutoff, tau, kernel_type, power):
        super().__init__(cutoff, tau, kernel_type, power)
        self.Kernel = KernelFunction(self.kernel_type, self.power, self.tau)
        self.cutoff = cutoff
        self.atom_lig = atoms_lig
        self.atom_pro = atoms_pro
        self.all_pairs = self.get_pairs()
        self.pairwise_atom_type_radii = self.get_pairwise_atom_type_radii(self.all_pairs)
    def get_pairs(self):
        atoms_pair = []
        for pro in self.atom_pro:
            for lig in self.atom_lig:
                atoms_pair.append(f"{pro}(pro)-{lig}(lig)")
            for pro2 in self.atom_pro:
                atoms_pair.append(f"{pro}(pro1)-{pro2}(pro2)")
        for lig1 in self.atom_lig:
            for lig2 in self.atom_lig:
                atoms_pair.append(f"{lig1}(lig1)-{lig2}(lig2)")
        return atoms_pair
    def get_pairwise_atom_type_radii(self, atom_pairs):
        atom_radii_dict = {a: radius_map[a] for a in self.atom_lig}
        pairwise_atom_type_radii = {}
        for i in atom_pairs:
            atom1 = (i.split('-')[0]).split('(')[0]
            atom2 = (i.split('-')[1]).split('(')[0]
            pairwise_atom_type_radii[i] = atom_radii_dict[atom1] + atom_radii_dict[atom2] 
        return pairwise_atom_type_radii
    def get_mwcg_rigidity(self, protein_file, ligand_file):
        protein = pdb_to_df(protein_file)
        ligand = mol2_to_df(ligand_file)
        for i in ["X", "Y", "Z"]:
            protein = protein[(protein[i] < float(ligand[i].max()) + self.cutoff) & 
                            (protein[i] > float(ligand[i].min()) - self.cutoff)]
        atom_pairs = []
        pair_types = []
        for p, l in product(protein["ATOM_ELEMENT"], ligand["ATOM_ELEMENT"]):
            atom_pairs.append(f"{p}(pro)-{l}(lig)")
            pair_types.append("pro-lig")
        for p1, p2 in product(protein["ATOM_ELEMENT"], repeat=2):
            atom_pairs.append(f"{p1}(pro1)-{p2}(pro2)")
            pair_types.append("pro-pro")
        for l1, l2 in product(ligand["ATOM_ELEMENT"], repeat=2):
            atom_pairs.append(f"{l1}(lig1)-{l2}(lig2)")
            pair_types.append("lig-lig")
        pairwise_radii = np.array([
            self.pairwise_atom_type_radii.get(x, 0)  
            for x in atom_pairs
        ])
        dist_pro_lig = cdist(protein[["X", "Y", "Z"]], ligand[["X", "Y", "Z"]], metric="euclidean")
        dist_pro_pro = cdist(protein[["X", "Y", "Z"]], protein[["X", "Y", "Z"]], metric="euclidean")
        dist_lig_lig = cdist(ligand[["X", "Y", "Z"]], ligand[["X", "Y", "Z"]], metric="euclidean")
        distances = np.concatenate([dist_pro_lig.ravel(), dist_pro_pro.ravel(), dist_lig_lig.ravel()])
        if len(pairwise_radii) != len(distances):
            raise ValueError(f"Mismatch between radii ({len(pairwise_radii)}) and distances ({len(distances)})")
        mwcg_distances = self.Kernel.kernel_function(distances, pairwise_radii)
        pairwise_mwcg = pd.DataFrame({"ATOM_PAIR": atom_pairs, "PAIR_TYPE": pair_types})
        mwcg_distances_df = pd.DataFrame({"DISTANCE": distances, "MWCG_DISTANCE": mwcg_distances})
        pairwise_mwcg = pd.concat([pairwise_mwcg, mwcg_distances_df], axis=1)
        pairwise_mwcg = pairwise_mwcg[pairwise_mwcg["DISTANCE"] <= self.cutoff].reset_index(drop=True)
        return pairwise_mwcg
    def get_pairwise_score(self, protein_file, ligand_file):
        features = ['SUM', 'MEAN', 'STD', 'MIN', 'MAX']
        pairwise_mwcg = self.get_mwcg_rigidity(protein_file, ligand_file)
        missing_pairs = [clu for clu in self.all_pairs if clu not in pairwise_mwcg['ATOM_PAIR'].values]
        if missing_pairs:
            new_rows = pd.DataFrame({'ATOM_PAIR': missing_pairs, 'MWCG_DISTANCE': [0] * len(missing_pairs)})
            pairwise_mwcg = pd.concat([pairwise_mwcg, new_rows], ignore_index=True)
        mwcg_temp = pairwise_mwcg.groupby('ATOM_PAIR', as_index=True)['MWCG_DISTANCE'].agg(['sum', 'mean', 'std', 'min', 'max']).fillna(0)
        mwcg_temp.columns = features
        pairwise_score = pd.DataFrame({'ATOM_PAIR': self.all_pairs})
        for _f in features:
            pairwise_score[_f] = 0
        pairwise_score = pairwise_score.set_index('ATOM_PAIR').add(mwcg_temp, fill_value=0).reset_index()
        return pairwise_score
class SYBYL_GGL(PARAMS):
    protein_atom_types_df = pd.read_csv(
        './utils/protein_atom_types.csv')
    ligand_atom_types_df = pd.read_csv(
        './utils/ligand_SYBYL_atom_types.csv')
    protein_atom_types = protein_atom_types_df['AtomType'].tolist()
    protein_atom_radii = protein_atom_types_df['Radius'].tolist()
    ligand_atom_types = ligand_atom_types_df['AtomType'].tolist()
    ligand_atom_radii = ligand_atom_types_df['Radius'].tolist()
    protein_ligand_atom_types = [
        i[0]+"-"+i[1] for i in product(protein_atom_types, ligand_atom_types)]
    def __init__(self, cutoff, tau, kernel_type, power):
        super().__init__(cutoff, tau, kernel_type, power)
        self.Kernel = KernelFunction(self.kernel_type, self.power, self.tau)
        self.pairwise_atom_type_radii = self.get_pairwise_atom_type_radii()
    def get_pairwise_atom_type_radii(self):
        protein_atom_radii_dict = {a: r for (a, r) in zip(
            self.protein_atom_types, self.protein_atom_radii)}
        ligand_atom_radii_dict = {a: r for (a, r) in zip(
            self.ligand_atom_types, self.ligand_atom_radii)}
        pairwise_atom_type_radii = {i[0]+"-"+i[1]: protein_atom_radii_dict[i[0]] +
                                    ligand_atom_radii_dict[i[1]] for i in product(self.protein_atom_types, self.ligand_atom_types)}
        return pairwise_atom_type_radii
    def mol2_to_df(self, mol2_file):
        df_mol2 = PandasMol2().read_mol2(mol2_file).df
        df = pd.DataFrame(data={'ATOM_INDEX': df_mol2['atom_id'],
                                'ATOM_ELEMENT': df_mol2['atom_type'],
                                'X': df_mol2['x'],
                                'Y': df_mol2['y'],
                                'Z': df_mol2['z']})
        if len(set(df["ATOM_ELEMENT"]) - set(self.ligand_atom_types)) > 0:
            print(
                "WARNING: Ligand contains unsupported atom types. Only supported atom-type pairs are counted.")
        return(df)
    def pdb_to_df(self, pdb_file):
        ppdb = PandasPdb()
        ppdb.read_pdb(pdb_file)
        ppdb_all_df = ppdb.df['ATOM']
        ppdb_df = ppdb_all_df[ppdb_all_df['atom_name'].isin(
            self.protein_atom_types)]
        atom_index = ppdb_df['atom_number']
        atom_element = ppdb_df['atom_name']
        x, y, z = ppdb_df['x_coord'], ppdb_df['y_coord'], ppdb_df['z_coord']
        df = pd.DataFrame.from_dict({'ATOM_INDEX': atom_index, 'ATOM_ELEMENT': atom_element,
                                     'X': x, 'Y': y, 'Z': z})
        return df
    def get_mwcg_rigidity(self, protein_file, ligand_file):
        '''
            Adapted from ECIF package
        '''
        protein = self.pdb_to_df(protein_file)
        ligand = self.mol2_to_df(ligand_file)
        for i in ["X", "Y", "Z"]:
            protein = protein[protein[i] < float(ligand[i].max())+self.cutoff]
            protein = protein[protein[i] > float(ligand[i].min())-self.cutoff]
        atom_pairs = list(
            product(protein["ATOM_ELEMENT"], ligand["ATOM_ELEMENT"]))
        atom_pairs = [x[0]+"-"+x[1] for x in atom_pairs]
        pairwise_radii = [self.pairwise_atom_type_radii[x]
                          for x in atom_pairs]
        pairwise_radii = np.asarray(pairwise_radii)
        pairwise_mwcg = pd.DataFrame(atom_pairs, columns=["ATOM_PAIR"])
        distances = cdist(protein[["X", "Y", "Z"]],
                          ligand[["X", "Y", "Z"]], metric="euclidean")
        pairwise_radii = pairwise_radii.reshape(
            distances.shape[0], distances.shape[1])
        mwcg_distances = self.Kernel.kernel_function(distances, pairwise_radii)
        distances = distances.ravel()
        mwcg_distances = mwcg_distances.ravel()
        mwcg_distances = pd.DataFrame(
            data={"DISTANCE": distances, "MWCG_DISTANCE": mwcg_distances})
        pairwise_mwcg = pd.concat([pairwise_mwcg, mwcg_distances], axis=1)
        pairwise_mwcg = pairwise_mwcg[pairwise_mwcg["DISTANCE"] <= self.cutoff].reset_index(
            drop=True)
        return pairwise_mwcg
    def get_ggl_score(self, protein_file, ligand_file):
        features = ['COUNTS', 'SUM', 'MEAN', 'STD', 'MIN', 'MAX']
        pairwise_mwcg = self.get_mwcg_rigidity(protein_file, ligand_file)
        mwcg_temp_grouped = pairwise_mwcg.groupby('ATOM_PAIR')
        mwcg_temp_grouped.agg(['sum', 'mean', 'std', 'min', 'max'])
        mwcg_temp = mwcg_temp_grouped.size().to_frame(name='COUNTS')
        mwcg_temp = (mwcg_temp
                     .join(mwcg_temp_grouped.agg({'MWCG_DISTANCE': 'sum'}).rename(columns={'MWCG_DISTANCE': 'SUM'}))
                     .join(mwcg_temp_grouped.agg({'MWCG_DISTANCE': 'mean'}).rename(columns={'MWCG_DISTANCE': 'MEAN'}))
                     .join(mwcg_temp_grouped.agg({'MWCG_DISTANCE': 'std'}).rename(columns={'MWCG_DISTANCE': 'STD'}))
                     .join(mwcg_temp_grouped.agg({'MWCG_DISTANCE': 'min'}).rename(columns={'MWCG_DISTANCE': 'MIN'}))
                     .join(mwcg_temp_grouped.agg({'MWCG_DISTANCE': 'max'}).rename(columns={'MWCG_DISTANCE': 'MAX'}))
                     )
        mwcg_columns = {'ATOM_PAIR': self.protein_ligand_atom_types}
        for _f in features:
            mwcg_columns[_f] = np.zeros(len(self.protein_ligand_atom_types))
        ggl_score = pd.DataFrame(data=mwcg_columns)
        ggl_score = ggl_score.set_index('ATOM_PAIR').add(
            mwcg_temp, fill_value=0).reindex(self.protein_ligand_atom_types).reset_index()
        return ggl_score
def mol2_to_df(mol2_file):
    df_mol2 = PandasMol2().read_mol2(mol2_file).df
    filtered_df_mol2 = df_mol2.loc[df_mol2['atom_type'].apply(lambda x: x.split('.')[0]).isin(atoms_lig)]
    df = pd.DataFrame({
        'ATOM_INDEX': filtered_df_mol2['atom_id'],
        'ATOM_ELEMENT': filtered_df_mol2['atom_type'].apply(lambda x: x.split('.')[0]),
        'X': filtered_df_mol2['x'],
        'Y': filtered_df_mol2['y'],
        'Z': filtered_df_mol2['z'],
        'radi':  filtered_df_mol2['atom_type'].apply(lambda x: radius_map.get(x.split('.')[0], np.nan))
    })
    return(df)
def pdb_to_df(pdb_file):
    ppdb = PandasPdb()
    ppdb.read_pdb(pdb_file)
    ppdb_all_df = ppdb.df['ATOM']
    ppdb_df = ppdb_all_df[ppdb_all_df['atom_name'].apply(lambda x: x.split('.')[0]).isin(atoms_pro)]
    atom_index = ppdb_df['atom_number']
    atom_element = ppdb_df['atom_name']
    x, y, z = ppdb_df['x_coord'], ppdb_df['y_coord'], ppdb_df['z_coord']
    df = pd.DataFrame.from_dict({'ATOM_INDEX': atom_index, 
                                 'ATOM_ELEMENT': atom_element,
                                 'X': x, 
                                 'Y': y, 
                                 'Z': z})
    return df
