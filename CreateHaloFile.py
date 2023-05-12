import numpy as np
from pygadgetreader import *
import h5py as h5
import os
import sys
from collections import defaultdict


def nested_dict(n, type):
    """Creates a nested dictionary. Will be used to create dictionaries with halo information."""
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))


def get_r200b(m200b, Hubble, Omega_m):
    """
    Computes radius of sphere within which the density is 200 times the background density of the universe.
    """
    G = 6.67259e-11               # Gravitational constant in m^3 / (kg s^2)
    M_sun = 1.989e30              # Solar mass in kg
    H = Hubble * 1000 / 3.086e22  # Hubble parameter in s^-1
    
    r200b = (G * m200b * M_sun / (100 * Omega_m * H**2))**(1/3)
    return r200b / 3.086e22 # To get r200b in units of Mpc


def create_dict_halos(file_path, n_halos, Hubble, Omega_m, all_halos=False, print_info=False):
    """
    Creates dictionary with halo properties at redshift z. The halo properties returned are:
    - HaloID   (Halo id)
    - rvir     (Virial radius)    
    - mvir     (Virial mass)
    - m200b    (Mass inside sphere with matter with density 200 times the background density of the universe)
    - pos      (Coordinates of halo center as vector, [x, y, z])
    - vel      (Vector with velocity components, [vx, vy, vz])
    
    Input:
        - file_path: Path to file where properties of all halos (from rockstar) are stored.
        - n_halos: Number of largest halos to be found based on virial mass.
        - Hubble: Hubble parameter at redshift of snapshot in which halos are found.
        - Omega_m: Matter density parameter at redshift of snapshot in which halos are found.
        - all_halos: Set to True if all halos are wanted. n_halos is then ignored.
    Output:
        - Dictionary containing the above properties for all halos or only the n largest halos. 
          The dictionary is on the form (for redshift z=0) name_of_dictionary["z=0"]["property"] = array(number_of_halos). 
          "property" is found above, f.ex. for virial radius, "property" = "rvir".
          The virial radius is returned in unit Mpc.
          The keys of the dictionary contains arrays also when only the largest halo is considered.
    """
    
    assert n_halos != 0, "n_halos must be 1 or larger, not 0. If all halos are needed set all_halos to True!"
    
    # Store header and value of scale factor to find redshift
    file = open(file_path, "r")
    header = np.array(file.readline()[1:].split())
    
    scale_factor = float(file.readline().split()[-1])
    redshift = (1 / scale_factor) - 1
    # Check if redshift, z, is integer to avoid having to z=0.0 as key in dictionary
    if redshift.is_integer():
        redshift = int(redshift)
    else:
        redshift = round(redshift, 1) # Round redshift to one decimal place
    # Read and print boxsize and number of grids ???
    file.close()

    redshift_key = "z=" + str(redshift) # First keyword of dictionary

    data = np.loadtxt(file_path) # Store values of halo properties 
        
    # Find indices of id, virial mass, virial radius, position and velocity components in header
    idx_id = np.where(header == "id")[0][0]
    idx_mvir = np.where(header == "mvir")[0][0]
    idx_m200b = np.where(header == "m200b")[0][0]
    idx_rvir = np.where(header == "rvir")[0][0]
    idx_x = np.where(header == "x")[0][0]
    idx_vx = np.where(header == "vx")[0][0]

    if all_halos == True:
        print(f"Create dictionary with properties of all {data.shape[0]} halos!")
        data_new = data
    
    # Keep only the properties of the n largest halos 
    else:
        (print("Create dictionary with properties of the largest halo!") if n_halos==1 else 
         print(f"Create dictionary with properties of the {n_halos} largest halos!"))

        # Get indices of n largest halos based on virial mass. 
        # Reverse to get list in descending order
        idx_sorted = np.argsort(data[:, idx_mvir])[::-1][:n_halos]
        data_new = data[idx_sorted, :]
        
    
    # Create dictionary with halo properties
    halo_dict = nested_dict(2, list)
    
    # Add id, virial mass, virial radius, position and velocity to dicitonary
    halo_dict[redshift_key]["haloID"] = data_new[:, idx_id].astype(int)
    halo_dict[redshift_key]["mvir"] = data_new[:, idx_mvir]
    halo_dict[redshift_key]["m200b"] = data_new[:, idx_m200b]
    halo_dict[redshift_key]["rvir"] = data_new[:, idx_rvir] / 1000 # Convert virial radii from kpc to Mpc    
    halo_dict[redshift_key]["r200b"] = get_r200b(halo_dict[redshift_key]["m200b"], Hubble, Omega_m)
    
    # Transpose arrays to get position and velocity as arrays with vectors, i.e. 
    # [[x1, y1, y1], ..., [xn, yn, zn]] and [[vx1, vy1, vy1], ..., [vxn, vyn, vzn]]
    halo_dict[redshift_key]["pos"] = np.array([data_new[:,idx_x], data_new[:,idx_x+1], data_new[:,idx_x+2]]).T
    halo_dict[redshift_key]["vel"] = np.array([data_new[:,idx_vx], data_new[:,idx_vx+1], data_new[:,idx_vx+2]]).T    
    
    if print_info == True:
        print("Number of halos: ", data.shape[0])
        print(f"Snapshot redshift: z={redshift}")
    
    return halo_dict


def create_halo_file(file_path, halo_dict, rad_max, output_path=None, cosmo_theory=None, print_info=False):
    """
    Save info of halo and particles inside rad_max*rvir of the halo in HDF5 file.
    Assumes the input file is in gadget format.
    Create directory called output in the directory where the file running the code is stored. 
    If the directory already exists, and it contains files made earlier with this code, these
    files are removed and the new are added. 
    
    One file is created for each halo in halo_dict. Keys of halo and particle properties:
    halo:                      particles:
    - haloID                   - PID
    - rvir                     - pos (3D)
    - mvir                     - vel (3D) 
    - m200b
    - pos (3D)
    - vel (3D)
    
    Input: 
        - halo_dict: Dictionary with properties of the largest halos.
        - file_path: Path to gadget file with particle properties from simulation snapshot.
        - rad_max: All particles closer than rvir*rad_max to the halo center will be stored,
                  i.e. for rad_max=10 info of all particles inside a distance of 10 virial radii from the halo center is saved.
        - cosmo_theory: Cosmology theory. This is only added to the name of the output folder to clearly difference between the outputs... 
        - print_info: If True, halo properties like virial mass, virial radius and number of particles for each halo 
                      are printed. 
    """
    print(f"Cosmology theory: {cosmo_theory}") 
    
    redshift = list(halo_dict.keys())[0]
    rvir_arr = halo_dict[redshift]["rvir"]
    r200b_arr = halo_dict[redshift]["r200b"]
    rad = rad_max * rvir_arr # Particles inside this distance from the halo center will be stored
    n_halos = len(rvir_arr)
    
    mvir_arr = halo_dict[redshift]["mvir"]
    m200b_arr = halo_dict[redshift]["m200b"]
    
    (print("Create file with properties of the largest halo!") if n_halos==1 else 
    print(f"Create files with properties of the {n_halos} largest halos!"))
    
    # Positions and velocities from snapshot represented with float32...
    id_pcls = readsnap(file_path, "pid", "dm")
    vel_pcls = readsnap(file_path, "vel", "dm") 
    pos_pcls = readsnap(file_path, "pos", "dm") / 1000 # Convert positions from kpc to Mpc
    pos_halo = halo_dict[redshift]["pos"]
    
    print("")
    if print_info == True:
        print("{:<10} | {} | {} | {} | {} | {:<10} |".format("", "Number of particles", "Virial radius (Mpc)", "r200b (Mpc)", "Virial mass (solar masses)", "m200b (Solar masses)"))
        width = 21+len("Number of particles")+len("Virial mass (solar masses)")+len("Virial radius (Mpc)")+len("m200b (Solar masses)")+len("r200b (Mpc)")+7
        print("-"*width)

    
    # How to save files to distinguish different cosmological simulations???
    # Use new filenames or change the name of the output folder???
    
    # Name of output directory
    if cosmo_theory == None:
        output_dir = f"output_halo_info"
    else:
        output_dir = f"output_halo_info_{cosmo_theory}"

    # Set output_path to path of current directory (where the code is run) if no output path was given
    if output_path == None:
        output_path = os.getcwd()

    path_output_dir = os.path.join(output_path, output_dir)

    # Create output directory if it does not already exist
    if not os.path.exists(path_output_dir):
        os.mkdir(path_output_dir)
        
    # Find particles inside rad_max*rvir of halos
    for i in range(n_halos):
        radial_vector = pos_pcls - pos_halo[i] # Vector from halo centre to particles
        r_pcls = np.linalg.norm(radial_vector, axis=1) # Distance from halo center to particles in Mpc, r_pcls = sqrt(x^2 + y^2 + z^2)
        idx_in = np.where(r_pcls < rad[i])[0]
        
        # Other ways of saving data, create directory, new directory if directory already exists,
        # directory name...
                    
        out_file = f"halo_{i+1}_{redshift}.h5"  # Name of output file
        
        path_out = os.path.join(path_output_dir, out_file)
        
        # Create directory for saving file, overwrite old file, add new file with new filename ???
        # delete and create new file ???
        if os.path.isfile(path_out):
            os.remove(path_out)
            
        hf = h5.File(path_out, "w")

        gh = hf.create_group("halo")
        for hparam in halo_dict[redshift].keys():
            gh.create_dataset(hparam, data=halo_dict[redshift][hparam][i])
    
        gp = hf.create_group("particles")
        gp.create_dataset("PID", data=id_pcls[idx_in])
        gp.create_dataset("pos", data=pos_pcls[idx_in])
        gp.create_dataset("vel", data=vel_pcls[idx_in])
        
        hf.close()
            
        if print_info == True:
            print("{:<10} | {:>19} | {:>19.3f} | {:>11.3f} | {:>26.2e} | {:>20.2e} |".format(f"Halo {i+1}", len(pos_pcls[idx_in]), rvir_arr[i], r200b_arr[i], mvir_arr[i], m200b_arr[i]))