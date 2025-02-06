
import numpy as np

class KernelFunction:

    def __init__(self, kernel_type='exponential_kernel',
                 kappa=2.0, tau=1.0):
        self.kernel_type = kernel_type
        self.kappa = kappa
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
        return np.exp(-((d / eta) ** self.kappa))

    def lorentz_kernel(self, d, sum_radii):
        eta = self.tau*sum_radii
        return (eta / (eta + d)) ** self.kappa


class RI_SCORE:
    def __init__(self, Kernel, cutoff):
        self.Kernel = Kernel
        self.cutoff = cutoff
    
    def calculate_ri(self, matrix1, matrix2):
        if matrix1 is None or matrix2 is None:
            return 0.0
        
        radiis1 = np.array([atom[-1] for atom in matrix1])
        radiis2 = np.array([atom[-1] for atom in matrix2])
        coords1 = np.array([atom[:3] for atom in matrix1])
        coords2 = np.array([atom[:3] for atom in matrix2])
        
        # Generate all pairwise indices
        idx1, idx2 = np.meshgrid(np.arange(len(coords1)), np.arange(len(coords2)), indexing="ij")
        idx1 = idx1.ravel()
        idx2 = idx2.ravel()
        
        # Compute distances and sum_radii (adjusted by tau)
        d = np.linalg.norm(coords1[idx1] - coords2[idx2], axis=1)
        sum_radii = (radiis1[idx1] + radiis2[idx2])
        
        # Apply cutoff and kernel function
        mask = d <= self.cutoff
        score = np.zeros_like(d)
        score[mask] = self.kernel.kernel_function(d[mask], sum_radii[mask])
        
        return float(np.sum(score))
    
    def ri_triplet(self, atom1s, atom2s, atom3s):
        if None in (atom1s, atom2s, atom3s):
            return 0.0
        
        score1 = self.calculate_ri(atom1s, atom2s)
        score2 = self.calculate_ri(atom2s, atom3s)
        score3 = self.calculate_ri(atom1s, atom3s)
        
        return float(score1 + score2 + score3)
    
    def calculate_all_cluster(self, pro_name_data, clusters):
        ri_scores = {}
        for clu in clusters:
            # Extract atom1
            atom1_name = clu['pro1']
            atom1s = pro_name_data['protein'].get(atom1_name, None)
            
            # Extract atom2 and atom3 based on cluster type
            atom2s, atom3s = None, None
            if 'pro2' in clu and 'lig' in clu:  # Protein-protein-ligand cluster
                atom2s = pro_name_data['protein'].get(clu['pro2'], None)
                atom3s = pro_name_data['ligand'].get(clu['lig'], None)
            elif 'lig1' in clu and 'lig2' in clu:  # Protein-ligand-ligand cluster
                atom2s = pro_name_data['ligand'].get(clu['lig1'], None)
                atom3s = pro_name_data['ligand'].get(clu['lig2'], None)
            
            # Calculate score for the current cluster
            ri_scores[str(clu)] = self.ri_triplet(atom1s, atom2s, atom3s)
        
        return ri_scores

class CLUSTER:
    def __init__(self, atoms_pro, atoms_lig):
        self.atoms_pro = atoms_pro
        self.atoms_lig = atoms_lig

    ### get cluster
    def get_cluster(self): 
        cluster = []
        for pro1 in self.atoms_pro:
            for pro2 in self.atoms_pro:
                for lig in self.atoms_lig:
                    if pro1 != pro2:
                        cluster.append({'pro1' :pro1, 'pro2': pro2,'lig' : lig})
        for pro1 in self.atoms_pro:
            for lig1 in self.atoms_lig:
                for lig2 in self.atoms_lig:
                    if lig1 != lig2:
                        cluster.append({'pro1' :pro1, 'lig1': lig1,'lig2' : lig2})
        return cluster
    
    def format_clustername(clustername):
        # Remove curly braces and quotes
        cluster_dict = eval(clustername)
        
        # Check if it's the first type (Protein-protein-ligand)
        if 'pro1' in cluster_dict and 'pro2' in cluster_dict and 'lig' in cluster_dict:
            formatted_name = f"{cluster_dict['pro1']}(pro)-{cluster_dict['pro2']}(pro)-{cluster_dict['lig']}(lig)"
        # Check if it's the second type (Protein-ligand-ligand)
        elif 'pro1' in cluster_dict and 'lig1' in cluster_dict and 'lig2' in cluster_dict:
            formatted_name = f"{cluster_dict['pro1']}(pro)-{cluster_dict['lig1']}(lig)-{cluster_dict['lig2']}(lig)"
        else:
            formatted_name = str(cluster_dict)  # In case of an unknown format
        
        return formatted_name


