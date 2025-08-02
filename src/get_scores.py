import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.spatial.distance import cdist
from itertools import product
from biopandas.pdb import PandasPdb
from biopandas.mol2 import PandasMol2
from scipy.spatial import  KDTree
from collections import defaultdict


atoms_lig = ['C','N','O','S','P','F','Cl','Br','I']
atoms_pro = ['C','N','O','S']

radius_map = { 'N': 1.55,
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

    def __init__(self, kernel_type='exponential_kernel',
                 power=2.0, tau=1.0):
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
    # def __init__(self):
    ### get cluster
    def get_cluster(self): 
        cluster = []
        for pro1 in atoms_pro:
            for pro2 in atoms_pro:
                for lig in atoms_lig:
                    # if pro1 != pro2:
                        cluster.append({'pro1' :pro1, 'pro2': pro2,'lig' : lig})
                for pro3 in atoms_pro:
                    cluster.append({'pro1' :pro1, 'pro2': pro2,'pro3' : pro3})
            for lig1 in atoms_lig:
                for lig2 in atoms_lig:
                    # if lig1 != lig2:
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


class TRIPLET_SCORE:
    def __init__(self, cutoff, stat, tau, kernel_type):
    
        self.Kernel = KernelFunction(kernel_type='l', power=5.0, tau=0.5)
        self.cutoff = cutoff
        self.atom_lig = atoms_lig
        self.atom_pro = atoms_pro
        self.stat = stat
        self.pairwise_atom_type_radii = self.get_pairwise_atom_type_radii()
        self.cluster = CLUSTER()
        self.tau = tau
        self.kernel_type = kernel_type

    def get_pairwise_atom_type_radii(self):
        protein_atom_radii_dict = {a: radius_map[a] for a in self.atom_pro}
        ligand_atom_radii_dict = {a: radius_map[a] for a in self.atom_lig}
        pairwise_atom_type_radii = {}
        for i in product(self.atom_pro, self.atom_lig):
            pairwise_atom_type_radii[f"{i[0]}-{i[1]}"] =  protein_atom_radii_dict[i[0]] + ligand_atom_radii_dict[i[1]] 
        for i in product(self.atom_pro, self.atom_pro):
            pairwise_atom_type_radii[f"{i[0]}-{i[1]}"] =  protein_atom_radii_dict[i[0]] + protein_atom_radii_dict[i[1]] 
        for i in product(self.atom_lig, self.atom_lig):
            pairwise_atom_type_radii[f"{i[0]}-{i[1]}"] =  ligand_atom_radii_dict[i[0]] + ligand_atom_radii_dict[i[1]] 

        return pairwise_atom_type_radii

    def get_triplet(self, protein_file, ligand_file):
        # Load data and convert to numpy arrays
        protein_df = pdb_to_df(protein_file)
        ligand_df = mol2_to_df(ligand_file)
        
        prot_coords = protein_df[['X', 'Y', 'Z']].values.astype(np.float32)
        prot_elements = protein_df['ATOM_ELEMENT'].values
        lig_coords = ligand_df[['X', 'Y', 'Z']].values.astype(np.float32)
        lig_elements = ligand_df['ATOM_ELEMENT'].values

        # Element handling
        unique_elements = sorted(set(prot_elements) | set(lig_elements))
        element_to_idx = {e: i for i, e in enumerate(unique_elements)}
        self.idx_to_element = {i: e for e, i in element_to_idx.items()}
        n_elements = len(unique_elements)
        
        # Precompute radii matrix
        radii_matrix = np.zeros((n_elements, n_elements), dtype=np.float32)
        for i in range(n_elements):
            for j in range(n_elements):
                e1, e2 = self.idx_to_element[i], self.idx_to_element[j]
                key = f"{e1}-{e2}"
                if key not in self.pairwise_atom_type_radii:
                    key = f"{e2}-{e1}"
                radii_matrix[i, j] = self.pairwise_atom_type_radii.get(key, 0.0)
        
        # Convert elements to indices
        prot_element_indices = np.array([element_to_idx[e] for e in prot_elements], dtype=np.int16)
        lig_element_indices = np.array([element_to_idx[e] for e in lig_elements], dtype=np.int16)

        # Spatial indices
        protein_tree = KDTree(prot_coords)
        ligand_tree = KDTree(lig_coords)
        
        # Score storage
        triplet_scores = defaultdict(list)
        kernel = self.Kernel.kernel_function
        cutoff = self.cutoff

        # Case 1: 2 protein + 1 ligand (prot-prot-lig)
        self._process_case(lig_coords, lig_element_indices, prot_coords, prot_element_indices, 
                        protein_tree, 'pro', 'pro', 'lig', radii_matrix, cutoff, kernel, triplet_scores)

        # Case 2: 1 protein + 2 ligand (prot-lig-lig)
        self._process_case(prot_coords, prot_element_indices, lig_coords, lig_element_indices,
                        ligand_tree, 'pro', 'lig', 'lig', radii_matrix, cutoff, kernel, triplet_scores, reverse=True)

        # Case 3: All protein (prot-prot-prot)
        self._process_homogeneous_case(prot_coords, prot_element_indices, protein_tree, 'pro', 
                                    radii_matrix, cutoff, kernel, triplet_scores)

        # Case 4: All ligand (lig-lig-lig)
        self._process_homogeneous_case(lig_coords, lig_element_indices, ligand_tree, 'lig',
                                    radii_matrix, cutoff, kernel, triplet_scores)

        # Handle cluster naming and empty results
        self._add_missing_clusters(triplet_scores)
        
        return self._create_final_dataframe(triplet_scores)


    def _process_case(self, primary_coords, primary_elements, secondary_coords, secondary_elements,
                        tree, primary_suffix, secondary_suffix1, secondary_suffix2, radii_matrix,
                        cutoff, kernel, triplet_scores, reverse=False):
        # Precompute element strings for fast lookup
        element_strings = [self.idx_to_element[i] for i in range(len(self.idx_to_element))]
        tau = self.tau  # Get tau from class instance

        # Process all primary atoms using vectorization
        for prim_idx in range(len(primary_coords)):
            prim_pos = primary_coords[prim_idx]
            prim_elem_idx = primary_elements[prim_idx]
            prim_elem_str = element_strings[prim_elem_idx]

            # Find nearby secondary atoms using spatial query
            secondary_neighbors = tree.query_ball_point(prim_pos, cutoff)
            if len(secondary_neighbors) < 2:
                continue

            # Extract neighbor data in bulk
            neighbor_coords = secondary_coords[secondary_neighbors]
            neighbor_elems = secondary_elements[secondary_neighbors]
            
            # Calculate all distances vectorized
            d_primary = np.linalg.norm(neighbor_coords - prim_pos, axis=1)
            dist_matrix = cdist(neighbor_coords, neighbor_coords)
            
            # Find valid pairs using triangular indices
            i, j = np.triu_indices(len(secondary_neighbors), k=1)
            mask = dist_matrix[i, j] <= cutoff
            valid_i, valid_j = i[mask], j[mask]
            
            if not valid_i.size:
                continue

            # Get element indices for valid pairs
            elem_i = neighbor_elems[valid_i]
            elem_j = neighbor_elems[valid_j]

            # Vectorized radius calculations
            if reverse:
                r1 = radii_matrix[elem_i, prim_elem_idx]
                r2 = radii_matrix[elem_j, prim_elem_idx]
            else:
                r1 = radii_matrix[prim_elem_idx, elem_i]
                r2 = radii_matrix[prim_elem_idx, elem_j]
            r3 = radii_matrix[elem_i, elem_j]

            # Calculate normalized distances with tau
            n1 = tau * r1
            n2 = tau * r2
            n3 = tau * r3

            # Get distances for valid pairs
            d1 = d_primary[valid_i]
            d2 = d_primary[valid_j]
            d3 = dist_matrix[valid_i, valid_j]

            # Vectorized kernel computation
            scores = kernel(d1, n1) + kernel(d2, n2) + kernel(d3, n3)

            # Bulk string generation
            elem_i_str = np.array(element_strings)[elem_i]
            elem_j_str = np.array(element_strings)[elem_j]
            
            triplet_types = [
                f"{prim_elem_str}({primary_suffix})-{e1}({secondary_suffix1})-{e2}({secondary_suffix2})"
                for e1, e2 in zip(elem_i_str, elem_j_str)
            ]

 
            # Update scores in bulk
            for t, s in zip(triplet_types, scores):
                triplet_scores[t].append(s)



    def _process_homogeneous_case(self, coords, element_indices, tree, suffix, 
                                radii_matrix, cutoff, kernel, triplet_scores):
        # Precompute element strings for fast lookup
        element_strings = [self.idx_to_element[i] for i in range(len(self.idx_to_element))]
        tau = self.tau  # Get tau from class instance

        # Batch process all atoms as centers
        for center_idx in range(len(coords)):
            center_pos = coords[center_idx]
            center_elem_idx = element_indices[center_idx]
            center_elem_str = element_strings[center_elem_idx]

            # Find all neighbors within cutoff
            neighbors = tree.query_ball_point(center_pos, cutoff)
            if len(neighbors) < 2:
                continue

            # Extract neighbor data in bulk
            neighbor_coords = coords[neighbors]
            neighbor_elems = element_indices[neighbors]
            
            # Calculate distances from center to neighbors
            d_center = np.linalg.norm(neighbor_coords - center_pos, axis=1)
            
            # Calculate pairwise distances between neighbors
            dist_matrix = cdist(neighbor_coords, neighbor_coords)
            
            # Find valid pairs using triangular indices
            i, j = np.triu_indices(len(neighbors), k=1)
            mask = dist_matrix[i, j] <= cutoff
            valid_i, valid_j = i[mask], j[mask]
            
            if not valid_i.size:
                continue

            # Get element indices for valid pairs
            elem_i = neighbor_elems[valid_i]
            elem_j = neighbor_elems[valid_j]

            # Vectorized radius calculations with tau
            r1 = radii_matrix[center_elem_idx, elem_i] * tau
            r2 = radii_matrix[center_elem_idx, elem_j] * tau
            r3 = radii_matrix[elem_i, elem_j] * tau

            # Get distances for valid pairs
            d1 = d_center[valid_i]
            d2 = d_center[valid_j]
            d3 = dist_matrix[valid_i, valid_j]

            # Vectorized kernel computation
            scores = kernel(d1, r1) + kernel(d2, r2) + kernel(d3, r3)

            # Bulk string generation
            elem_i_str = np.array(element_strings)[elem_i]
            elem_j_str = np.array(element_strings)[elem_j]
            
            # Create triplet types in bulk
            triplet_types = [
                f"{center_elem_str}({suffix})-{e1}({suffix})-{e2}({suffix})"
                for e1, e2 in zip(elem_i_str, elem_j_str)
            ]

            # Update scores in bulk
            for t, s in zip(triplet_types, scores):
                triplet_scores[t].append(s)


    def _add_missing_clusters(self, triplet_scores):
        """Ensure all predefined clusters are present in results"""
        cls = CLUSTER()
        cluster = cls.get_cluster()
        for clu in cluster:
            c = CLUSTER.format_clustername(clu)
            if c not in triplet_scores:
                triplet_scores[c] = [0]
        if not triplet_scores:
            triplet_scores['NA'] = [0]

    def _create_final_dataframe(self, triplet_scores):
        """Convert scores dictionary to final dataframe"""
        results = []
        for key, scores in triplet_scores.items():
            stats = {
                'SUM': np.sum(scores),
                'MIN': np.min(scores) if scores else 0,
                'MAX': np.max(scores) if scores else 0,
                'MEAN': np.mean(scores) if scores else 0,
                'STD': np.std(scores) if len(scores)>1 else 0,
                # 'COUNT': len(scores) if len(scores)>1 else 0
            }
            results.append({'type': key, **stats})
        return pd.DataFrame(results)
         

class PAIRWISE_SCORE:
    protein_ligand_atom_types = [
        i[0]+"-"+i[1] for i in product(atoms_lig, atoms_pro)]

    def __init__(self, Kernel, cutoff):

        self.Kernel = Kernel
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

        # Select protein atoms within cutoff distance from ligand
        for i in ["X", "Y", "Z"]:
            protein = protein[(protein[i] < float(ligand[i].max()) + self.cutoff) & 
                            (protein[i] > float(ligand[i].min()) - self.cutoff)]

        # Generate atom pairs
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

        # Lookup pairwise radii
        pairwise_radii = np.array([
            self.pairwise_atom_type_radii.get(x, 0)  # Use `.get(x, 0)` to avoid KeyErrors
            for x in atom_pairs
        ])

        # Compute distances
        dist_pro_lig = cdist(protein[["X", "Y", "Z"]], ligand[["X", "Y", "Z"]], metric="euclidean")
        dist_pro_pro = cdist(protein[["X", "Y", "Z"]], protein[["X", "Y", "Z"]], metric="euclidean")
        dist_lig_lig = cdist(ligand[["X", "Y", "Z"]], ligand[["X", "Y", "Z"]], metric="euclidean")

        # Flatten distance matrices
        distances = np.concatenate([dist_pro_lig.ravel(), dist_pro_pro.ravel(), dist_lig_lig.ravel()])
        
        # Ensure radii and distances are aligned
        if len(pairwise_radii) != len(distances):
            raise ValueError(f"Mismatch between radii ({len(pairwise_radii)}) and distances ({len(distances)})")

        # Apply kernel function
        mwcg_distances = self.Kernel.kernel_function(distances, pairwise_radii)

        # Create DataFrame
        pairwise_mwcg = pd.DataFrame({"ATOM_PAIR": atom_pairs, "PAIR_TYPE": pair_types})
        mwcg_distances_df = pd.DataFrame({"DISTANCE": distances, "MWCG_DISTANCE": mwcg_distances})

        # Concatenate atom pairs with distances
        pairwise_mwcg = pd.concat([pairwise_mwcg, mwcg_distances_df], axis=1)

        # Apply cutoff filter
        pairwise_mwcg = pairwise_mwcg[pairwise_mwcg["DISTANCE"] <= self.cutoff].reset_index(drop=True)

        return pairwise_mwcg


    def get_pairwise_score(self, protein_file, ligand_file):
        features = ['SUM', 'MEAN', 'STD', 'MIN', 'MAX']
        pairwise_mwcg = self.get_mwcg_rigidity(protein_file, ligand_file)
        # Ensure all 'ATOM_PAIR' are in the DataFrame
        missing_pairs = [clu for clu in self.all_pairs if clu not in pairwise_mwcg['ATOM_PAIR'].values]
        if missing_pairs:
            new_rows = pd.DataFrame({'ATOM_PAIR': missing_pairs, 'MWCG_DISTANCE': [0] * len(missing_pairs)})
            pairwise_mwcg = pd.concat([pairwise_mwcg, new_rows], ignore_index=True)

        # Group and aggregate
        mwcg_temp = pairwise_mwcg.groupby('ATOM_PAIR', as_index=True)['MWCG_DISTANCE'].agg(['sum', 'mean', 'std', 'min', 'max']).fillna(0)

        # Rename columns properly
        mwcg_temp.columns = features

        # Prepare ggl_score DataFrame with all_pairs
        pairwise_score = pd.DataFrame({'ATOM_PAIR': self.all_pairs})
        for _f in features:
            pairwise_score[_f] = 0

        # Merge or add values correctly
        pairwise_score = pairwise_score.set_index('ATOM_PAIR').add(mwcg_temp, fill_value=0).reset_index()

        return pairwise_score


class SYBYL_GGL:
    
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

    def __init__(self, Kernel, cutoff):

        self.Kernel = Kernel
        self.cutoff = cutoff

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

        # select protein atoms in a cubic with a size of cutoff from ligand
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



if __name__=='__main__':
    ri_instance = TRIPLET_SCORE(6.0, False, 0.5, 'l')
    re = ri_instance.get_triplet("/media/bio03/DATA1/tram/MD/Thesis/FRI_Advance/data/2016/CASF-2016/coreset/1a30/1a30_protein.pdb",'/media/bio03/DATA1/tram/MD/Thesis/FRI_Advance/data/2016/CASF-2016/coreset/1a30/1a30_ligand.mol2')
    re.to_csv("test.csv")