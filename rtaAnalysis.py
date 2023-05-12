import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import h5py as h5
from IPython.display import display, Latex
from scipy.optimize import curve_fit
import os
import sys
import re

plt.rc('figure', figsize=(8, 5))
plt.rc('font', size=15)
plt.rcParams['text.usetex'] = True

#@[]{}$

class turn_around_radius_analysis:

    def __init__(self, Hubble_today, n_phi, n_theta, n_shell, rad_max, dv=100, rad_in=3, rad_out=7, fitting_profile_threshold=0.1, rad_fit_list=[3,5,7,9], output_path=None, redshift=None, cosmo_theory=None, exclude_directions=True, use_fitting_threshold=True, path_background_file=None):
        
        assert n_shell > 0, f"The number of shells, n_shell, must be 1 or larger, not 0!"
        assert n_phi > 0, "n_phi must be 1 or larger, not 0!"
        assert n_theta > 0, "n_theta must be 1 or larger, not 0!"
        assert redshift != None, "The redshift in question must be given as input!"
        assert cosmo_theory != None, "The cosmological theory in question must be given as input!"
        assert path_background_file != None, "The path to the background file must be specified!"
        
        # Define input variables
        self.n_phi = n_phi         # Number of cones to make in phi (azimuth angle) direction 
        self.n_theta = n_theta     # Number of cones to make in theta (inclination angle) direction
        self.n_shell = n_shell     # Number of shells to make in radial direction
        self.rad_max = rad_max     # Number of virial radii inside which particles will be considered
        self.dv = dv               # Velocity offset when finding turnaround radius (use particles with velocities 0 +- dv)
        self.H0 = Hubble_today     # Hubble parameter at redshift z
#        self.Omega_m = Omega_m    # Density parameter for matter at redshift z
        self.fitting_profile_threshold = fitting_profile_threshold
        self.rad_fit_list = rad_fit_list                    # List with radii for performing fits used to exclude directions
        self.exclude_directions = exclude_directions        # Determines whether the directions are excluded or not
        self.use_fitting_threshold = use_fitting_threshold  # Determines if the fitting threshold is used to exclude directions or not
        
        # A power-law will be fitted to the velocity profiles between rad_in and rad_out times the virial radius
        self.rad_in = rad_in
        self.rad_out = rad_out
        
        # Path to directory where a directory with output files should be stored 
        self.output_path = output_path
        
        self.cosmo_theory = cosmo_theory
        self.redshift = redshift
        print(f"Cosmo theory: {cosmo_theory}")
        print(f"Redshift: z = {redshift}")
        
        # Creates output directory if not already created from a previous run
        self.make_output_dir()
        
        # Interpolate the Hubble parameter from the simulation output to get the Hubble parameter at the given redshift
        self.get_Hubble(path_background_file)
        
        print(f"Split halo/halos into {self.n_shell} parts in radial direction and {self.n_phi*self.n_theta} parts in the angular directions!") 


    def make_output_dir(self):
        """ 
        Create output directory if it does not exist
        """
        
        if self.output_path == None:
            output_path = os.getcwd() # Current folder
        else:
            output_path = self.output_path
        
        if self.cosmo_theory == None:
            output_file = "output_analysis"
        else:
            output_file = f"output_analysis_{self.cosmo_theory}"
        
        path_out = os.path.join(output_path, output_file)

        if not os.path.exists(path_out):
            os.mkdir(path_out)
            
        self.path_out = path_out
        

    def parse_parameter_file(self, file_path):
        """
        Reads settings file (from simulation or rockstar), take out properties needed 
        and make their values class attributes.
        
        Input:
            - file_path: Path to settings file given as string.
        """
        
        with open(file_path, "r") as f:
            for line in f:
                if "=" in line:
                    line_ = line.split("=")
                    param = line_[0].strip()
                    val = line_[1]
                    
                    # Create attributes with values of Hubble parameter h, boxsize and number of grids in smiulation
                    if param == "h":
                        n_decimals = len(val.split(".")[1])
                        val_ = float(val) * 100 # Multiply with 100 to get H in km/s/Mpc
                    elif param == "Ngrid":
                        self.n_grids = int(val)
                    elif param == "boxsize":
                        self.boxsize = float(val.split()[0])

#                    elif param == "omega_b":
#                        self.omega_b = float(val)
#                    elif param == "omega_cdm":
#                        self.omega_cdm = float(val)
                    
#                    elif param == "T_cmb":
#                        self.T_cmb = float(val.split()[0])

#                    elif param == "N_ur":
#                        self.N_ur = float(val)
                                                                                        
    
#    def get_r200b(self):
#        """
#       Computes the radius of a spherical region within which the average density is 200 times the background density of the universe.
#        Here the background density is taken to only be the background density of matter, rho_b = Omega_m * rho_critical. 
#        """
#        G = 6.67259e-11                    # Gravitational constant in m^3 / (kg s^2)
#        M_sun = 1.989e30                   # Solar mass in kg
#        H = self.Hubble * 1000 / 3.086e22  # Hubble parameter in s^-1
#        self.r200b = (G * self.m200b * M_sun / (100 *self.Omega_m * H**2))**1/3

    
    def read_halo_file(self, file_path):
        """
        Define variables for halo and particle properties needed.
        This function has to be run first in order for the analysis to be performed.
        """
        
        self.halo_file_path = file_path
        
        # Get number of halo from filename
        filename = file_path.split("/")[-1]
        self.n_halo = int(re.findall(r"\d+", filename)[0])
        sort_param = filename.split(".")[0].split("_")[-1]
        
        file = h5.File(file_path, "r")
        self.rvir = np.array(file["halo"]["rvir"])
        self.mvir = np.array(file["halo"]["mvir"])
        self.r200b = np.array(file["halo"]["r200b"])
        self.m200b = np.array(file["halo"]["m200b"])
        self.pos_halo = np.array(file["halo"]["pos"])
        self.vel_halo = np.array(file["halo"]["vel"])
        self.pos_pcls = np.array(file["particles"]["pos"])
        self.vel_pcls = np.array(file["particles"]["vel"])
        file.close()
        
#        self.get_r200b() # Compute the radius inside which the density is 200 times the background density
        
        
    def check_rad_pcls(self):
        if hasattr(self, "rad_pcls") == False:
            #print("rad_pcls was not defined!")
            self.get_rad_pcls()

    def check_rad_shell(self):
        if hasattr(self, "rad_shell") == False:
            #print("rad_shell was not defined!")
            self.get_rad_shell()
            
    def check_rad_shell_mean(self):
        if hasattr(self, "rad_shell_mean") == False:
            #print("rad_shell_mean was not defined!")
            self.get_rad_shell_mean()
            
    def check_angles_pcls(self):
        if (hasattr(self, "phi_pcls") == False) or (hasattr(self, "theta_pcls") == False):
            #print("angles_pcls was not defined!")
            self.get_angles_pcls()

    def check_angles_cones(self):
        if (hasattr(self, "phi_cone") == False) or (hasattr(self, "theta_cone") == False):
            #print("angles_cones was not defined!")
            self.get_angles_cones()
    
    def check_velocities_all(self):
        if (hasattr(self, "v_pec_all") == False) or (hasattr(self, "v_tot_all") == False):
            #print("v_pec_all and v_tot_all was not defined!")
            self.get_vel_pcls()
    
    def check_velocities_cones(self):
        if (hasattr(self, "v_pec_cones") == False) or (hasattr(self, "v_tot_cones") == False):
            #print("v_pec_cones and v_tot_cones was not defined!")
            self.get_vel_cones()
            
    def check_splits(self):
        if (hasattr(self, "idxs_arr_rad") == False) or (hasattr(self, "idxs_arr_angles") == False):
            self.split_rad()
            self.split_angles()
            
    def check_excluded_directions(self):
        if (hasattr(self, "idxs_exc_vel") == False) or (hasattr(self, "idxs_exc_coeff") == False):
            #print("Arrays with indices to be excluded was not defined!")
            self.pos_vel_dirs()
            self.neg_coeff_dirs()     
            
    def check_pos_vel_dirs(self):
        if (hasattr(self, "idxs_exc_vel") == False):
            self.pos_vel_dirs()
        
            
    def get_rad_pcls(self):
        #print("get_rad_pcls")
        """
        Create array with distances from particles to halo center.

        Output:
            - rad_pcls: Array with distances from halo center to particles in Mpc. 
        """
        
        assert hasattr(self, "pos_pcls") == True, "read_hdf5_file has to be run first to define the halo and particle properties!"
        
        self.rad_pcls = np.linalg.norm(self.pos_pcls - self.pos_halo, axis=1)
        return self.rad_pcls

    
    def get_angles_pcls(self):
        #print("get_angles_pcls")
        
        """
        Create arrays with azimuth angle phi and inclination angle theta of particles.

        Output:
            - phi_pcls: Array with azimuth angles of particles.
            - theta_pcls: Array with inclination angles of particles. 
        """

        assert hasattr(self, "pos_pcls") == True, "read_hdf5_file has to be run first to define the halo and particle properties!"
        
        self.check_rad_pcls() # Run get_rad_pcls() to get rad_pcls if rad_pcls is not defined
        
        x = self.pos_pcls[:, 0] - self.pos_halo[0]
        y = self.pos_pcls[:, 1] - self.pos_halo[1]
        z = self.pos_pcls[:, 2] - self.pos_halo[2]

        args = z / self.rad_pcls
        
        # Check if the argument that will be used in arccos is outside the interval [-1, 1] due to floating-point errors.
        # If the argument is outside this interval, the argument is set to 1 or -1.
        idxs_pos = np.where(args > 1)[0]
        idxs_neg = np.where(args < -1)[0]
        
        if len(idxs_pos) > 0 and len(idxs_neg) > 0:
            args[idxs_pos] = 1
            args[idxs_neg] = -1
        elif len(idxs_pos) > 0:
            args[idxs_pos] = 1
        elif len(idxs_neg) > 0:
            args[idxs_neg] = -1

        self.phi_pcls = np.arctan2(y, x)                # Azimuth angle phi of the particles
        self.theta_pcls = np.arccos(args)  # Inclination angle theta of particles

        return self.phi_pcls, self.theta_pcls
        
    
    def get_Hubble(self, path_background_file):
        """
        Compute the Hubble parameter at a given redshift by interpolating the Hubble parameter from the background file from gevolution.
        """
        data = np.loadtxt(path_background_file)
        a = data[:,2]          # Scale factor
        H_over_H0 = data[:,3]

        # Compute Hubble parameter H for each scale factor a
        H = H_over_H0 * self.H0 / a
        
        # Compute the scale factor at the given redshit
        a_z = 1 / (1 + self.redshift)
        
        # Interpolates the Hubble parameter at the given redshift
        self.Hubble = np.interp(a_z, a, H)

    
    def get_vel_pcls(self):
        #print("get_vel_pcls")
        
        """
        Create arrays with peculiar (velocities from simulation snapshot) and total velocities 
        of the particles in radial direction for one specific halo. 
        The total velocity is found as the sum of the peculiar and Hubble velocity.
        
        Output:
            - v_pec_rad: Array with peculiar velocities in radial direction for each particle in the halo.
            - v_tot: Array with the total velocities of the particles in radial direction, 
                     v_tot = v_pec_rad + v_Hubble.
        """
        
        assert hasattr(self, "pos_pcls") == True, "read_hdf5_file has to be run first to define the halo and particle properties!"
        
        # Change this...
        assert hasattr(self, "H0") == True, "parse_parameter_file has to be run to define the present Hubble parameter!"
        
        self.check_rad_pcls() # Run get_rad_pcls() to get rad_pcls if rad_pcls is not defined
        
        rad_vec_pcls = self.pos_pcls - self.pos_halo  # Particle position wrt halo center
        v_pec = self.vel_pcls - self.vel_halo         # Particle velocity wrt halo center

        n_vec = rad_vec_pcls / self.rad_pcls.reshape(-1, 1) # Normalized vector in radial direction 

        # Peculiar velocity of particles in radial direction wrt halo center
        # Dot product between peculiar velocity vector and normal vector in radial direction for each particle
        self.v_pec_all = np.sum(v_pec * n_vec, axis=1)
        
        # Find total velocity in radial direction, v_tot = v_Hubble + v_peculiar
        v_H = self.rad_pcls * self.Hubble             # Hubble/expansion velocity
        self.v_tot_all = v_H + self.v_pec_all

        return self.v_pec_all, self.v_tot_all
    
    
    def get_idxs_rta(self, outside_2rvir=False):
        #print("get_idxs_rta")
        """
        Find indices of particles with total velocity in a velocity interval 0 +- dv.
        Input:
            - outside_2rvir: Set to True to only consider particles outside 2 virial radii.

        Output: Return list with indices of particles with total velocity between +dv and -dv.
        """
        
        assert hasattr(self, "pos_pcls") == True, "read_hdf5_file has to be run first to define the halo and particle properties!"
        
        print(f"Find indices of particles at turnaround radius for Halo {self.n_halo}")
        
        self.check_rad_pcls()
        self.check_velocities_all()
        
        if outside_2rvir==True:
            # Find particles at turnaround radius and outside 2*rvir
            idx_outside = np.where(self.rad_pcls > 2*self.rvir)[0]
            v_tot_out = self.v_tot_all[idx_outside]
            idxs_vel_offset = np.where((v_tot_out > -self.dv) & (v_tot_out < self.dv))[0]
            self.idxs_rta = idx_outside[idxs_vel_offset]
        else:
            # Find all particles at turnaround radius, also inside 2*rvir
            self.idxs_rta = np.where((self.v_tot_all > -self.dv) & (self.v_tot_all < self.dv))[0]
        
        return self.idxs_rta
        
    
    def save_idxs_rta(self, outside_2rvir):
        """Save the indices of the particles at turnaround radius to already existing files containing halo and particle info."""
        
        self.get_idxs_rta(outside_2rvir)

        hf = h5.File(self.halo_file_path, "r")
        key_list = list(hf["particles"])
        hf.close()

        if "idxs_rta" in key_list:
            hf = h5.File(self.halo_file_path, "r+")
            data = hf["particles"]["idxs_rta"]
            data[...] = self.idxs_rta
            hf.close()
        else:
            hf = h5.File(self.halo_file_path, "a")
            hf.create_dataset("particles/idxs_rta", data=self.idxs_rta)
            hf.close()
    
        
    def get_rad_shell(self):
        #print("get_rad_shell")
        """
        Create array with inner and outer radii of shells.
        The array has length n_shell+1.
        """
        # Use logarithmically spaced radii ???
        self.rad_shell = np.linspace(0, self.rad_max*self.rvir, self.n_shell+1)
        return self.rad_shell

    
    def get_angles_cones(self):
        #print("get_angles_cones")
        """
        Create arrays with angles phi (azimuth) and theta (inclination) of cones/equal area segments.
        The lengths of the arrays are n_phi+1 and n_theta+1 since n_phi and n_theta are the 
        number of splits in the phi and theta direction.
        """

        # Array with angles phi used to split halo in azimuthal direction (e.g. in the xy-plane) 
        self.phi_cone = np.linspace(-np.pi, np.pi, self.n_phi+1) # +1 since n_phi is the number of intervals

        # Divide cones into equal area parts using the inclination/polar angle theta
        self.theta_cone = np.zeros(self.n_theta+1)  # List for storing theta values, +1 since n_theta is the number of intervals
        theta_init = 0                         # Initial theta

        # Use iterative method to find value of theta
        # Method taken from https://notmatthancock.github.io/2017/12/26/regular-area-sphere-partitioning.html
        for i in range(1, self.n_theta+1):
            self.theta_cone[i] = np.arccos(np.cos(theta_init) - i * 2/self.n_theta)

        return self.phi_cone, self.theta_cone

    
    def split_rad(self):
        #print("split_rad")
        """
        Split halo into n_shell spherical shells and find indices of 
        the particles in each shell.
 
        Output:
            - Array containing n_shell arrays with indices of the particles in each shell.
        """

        assert hasattr(self, "pos_pcls") == True, "read_hdf5_file has to be run first to define the halo and particle properties!"
                
        self.get_rad_pcls()
        self.get_rad_shell()

        idx_list = [] # List for storing indices of particles in shells

        # Find indices of particles in each shell
        for i in range(self.n_shell):
            idx_in_shell = np.where((self.rad_pcls >= self.rad_shell[i]) & (self.rad_pcls < self.rad_shell[i+1]))[0]
            idx_list.append(idx_in_shell)

        self.idxs_arr_rad = np.array(idx_list, dtype=np.ndarray)
        return self.idxs_arr_rad

    
    def split_angles(self): 
        #print("split_angles")
        """
        Split the halo into n_phi * n_theta equal area parts.
        Find indices of particles in inside each part. 

        Output:
            - 2D array with size n_phi * n_theta containing the indices of the particles
              inside each equal area part.
        """
        
        assert hasattr(self, "pos_pcls") == True, "read_hdf5_file has to be run first to define the halo and particle properties!"
        
        self.get_angles_cones() # Get phi and theta of cones
        self.get_angles_pcls() # Get phi and theta of particles
        
        idx_list = [[] for _ in range(self.n_phi)] # Create lists for storing indices of particles

        # Find particles belonging to the different cones and add their indices to list
        for i in range(self.n_phi):
            idx_in_phi = np.where((self.phi_pcls >= self.phi_cone[i]) & (self.phi_pcls < self.phi_cone[i+1]))[0] # Find indices of particles in phi intervals

            theta_temp = self.theta_pcls[idx_in_phi] # Get values of theta for particles in phi intervals
            for j in range(self.n_theta):
                # Find indices of particles in theta interval and phi interval
                idx_in_theta = np.where((theta_temp >= self.theta_cone[j]) & (theta_temp < self.theta_cone[j+1]))[0]
                idx_list[i].append(idx_in_phi[idx_in_theta]) # Get indices of particles between theta_j and theta_j+1 and between phi_i and phi_i+1

        self.idxs_arr_angles = np.array(idx_list, dtype=np.ndarray)
        return self.idxs_arr_angles

    
    def mean_all_pcls(self):
        #print("mean_all_pcls")
        """
        Compute mean and standard deviation of particles if n_phi, n_theta and n_shell are all 1,
        which means all particles are considered. Stops code since no analysis of the dependence
        of direction of particle velocities and turnaround radius can be performed when the halo
        is not split.
        """

        assert hasattr(self, "vel_pcls") == True, "read_hdf5_file has to be run first to define the halo and particle properties!"        
        
        self.check_velocities_all()
        
        print("Peculiar and total velocity for n_shell, n_phi and n_theta all equal to 1!")

        v_pec_mean = np.mean(self.v_pec_all)
        v_tot_mean = np.mean(self.v_tot_all)
        v_pec_std = np.sqrt(np.var(self.v_pec_all))
        v_tot_std = np.sqrt(np.var(self.v_tot_all))

        display(Latex("Average peculiar velocity for all particles: {:.3f} $\pm$ {:.3f} km/s.".format(v_pec_mean, v_pec_std)))
        display(Latex("Average total velocity for all particles: {:.3f} $\pm$ {:.3f} km/s.".format(v_tot_mean, v_tot_std)))

        assert self.n_shell!=1 and self.n_phi!=1 and self.n_theta!=1, "No velocity arrays are returned when n_shell, n_phi and n_theta are all 1!"

    
    def get_vel_cones(self):
        #print("get_vel_cones")
        """
        Create arrays with mean peculiar and total velocity of particles in each "cone". 
        Also create arrays with standard deviation for peculiar and total velocities in each "cone".

        Output:
            - v_pec_cones: Array with peculiar velocities for each cone.
            - v_tot_cones: Array with total velocities for each cone.
            - v_pec_std: Array with standard deviation of peculiar velocities for each cone.
            - v_tot_std: Array with standard deviation of total velocities for each cone.
            All output arrays have shape (n_phi, n_theta, n_shell).
            If all n_phi, n_theta, n_shell are 1, the output is only the mean 
            and standard deviations for all particle velocities.
        """
        
        assert hasattr(self, "vel_pcls") == True, "read_hdf5_file has to be run first to define the halo and particle properties!"
        
        n_phi = self.n_phi
        n_theta = self.n_theta
        n_shell = self.n_shell
        
        # Get indices of particles in shells and cones
        self.split_rad()
        self.split_angles()
                
        if n_shell==1 and n_phi==1 and n_theta==1:
            self.mean_all_pcls()
            
        # Find verage velocity in each shell for all cones 
        else:
            self.get_vel_pcls()
            self.v_pec_cones, self.v_tot_cones, self.std_pec, self.std_tot = [np.zeros([n_phi, n_theta, n_shell]) for _ in range(4)]

            # Array for storing the number of particles in each "halo part"
            self.num_count = np.zeros([n_phi, n_theta, n_shell])
            
            n_empty = 0
            # Find indices which are both in the cones and radial shells
            for i, arrays in enumerate(self.idxs_arr_angles):
                for j, idxs_ang in enumerate(arrays):
                    for k, idxs_rad in enumerate(self.idxs_arr_rad):
                        idxs_equal = np.nonzero(np.in1d(idxs_ang, idxs_rad))
                        idxs = idxs_ang[idxs_equal]
                        
                        # Skips shells with no particles. The value of the velocity in these shells is set to zero
                        if len(idxs) > 0:
                            self.v_pec_cones[i][j][k] = np.mean(self.v_pec_all[idxs])
                            self.v_tot_cones[i][j][k] = np.mean(self.v_tot_all[idxs])
                            self.std_pec[i][j][k] = np.std(self.v_pec_all[idxs]) # Standard deviation???
                            self.std_tot[i][j][k] = np.std(self.v_tot_all[idxs])
                            self.num_count[i][j][k] = len(idxs)
                        else:
                            n_empty += 1

            print("Empty shells: ", n_empty)
                        
            return self.v_pec_cones, self.v_tot_cones, self.std_pec, self.std_tot        
    
    
    def pos_vel_dirs(self):
        #print("pos_vel_dirs")
        """
        Find the cones (phi and theta) for which the peculiar velocity is positive 
        for radii larger than 2 virial radii. 

        Output:
            - Three arrays are output. These consist of:
                - The indices of the cones with positive velocities outside 2 virial radii
                  in phi and theta direction. The array is on the form [[idx_phi1, idx_theta1], [idx_phi2, idx_theta2], ...].
                - Two arrays which contains the max and min phi and theta of each cone with positive velocities
                  outside 2 virial radii. The arrays are on the form [[phi1_min, phi1_max], [phi2_min, phi2_max], ...] (similar for theta).
        """

        assert hasattr(self, "vel_pcls") == True, "read_hdf5_file has to be run first to define the halo and particle properties!"
        
        self.check_velocities_cones()
                
        idxs_pos_vel = np.where(self.v_pec_cones > 0) # Get indices (cones and shell) of particles with positive velocities

        idxs_out = np.where(self.rad_shell[idxs_pos_vel[2]]/self.rvir > 2)[0] # Get indices of particles outside 2 virial radii with positive velocities

        # Indices (in phi and theta direction) of particles outside 2 virial radii with positive velocities
        idxs_phi = idxs_pos_vel[0][idxs_out]
        idxs_theta = idxs_pos_vel[1][idxs_out]
        
        # Array with indices of directions with positive velocities. Transpose to get on desired form
        # This array might contain several equal arrays
        idxs_exc_all = np.array([idxs_phi, idxs_theta]).T

        # Lists for storing only one of each index combination and the phi and theta corresponding to each cone
        idxs_exc = []
        phi_exc = []
        theta_exc = []
        
        # Make lists with arrays with indices for each direction with positive velocities
        for i, j in np.ndindex(self.n_phi, self.n_theta):
            idxs = np.array([i, j])
            n_equal = np.where(np.all(idxs == idxs_exc_all, axis=1))[0] 
            if len(n_equal) > 0:
                idxs_exc.append(idxs)
                phi_exc.append([self.phi_cone[i], self.phi_cone[i+1]])
                theta_exc.append([self.theta_cone[j], self.theta_cone[j+1]])
        
        # The arrays phi_exc and theta_exc could still contain several equal arrays. The reason is that the first 
        # arrays in phi_exc and theta_exc belongs to the same cone. Thus, for example it could be that there are 
        # several cones with positive velocities in the same phi interval which means there would be several equal 
        # arrays in phi_exc but the corresponding arrays in theta_exc would be different. However, the combination 
        # of the n'th array in phi_exc and and the n'th array in theta_exc will never be the same.
            
        #print("Directions with positive peculiar velocities outside 2 virial radii: ", len(idxs_exc))
        
        self.idxs_exc_vel = np.array(idxs_exc)
        self.phi_exc_vel = np.array(phi_exc)
        self.theta_exc_vel = np.array(theta_exc)

        return self.idxs_exc_vel, self.phi_exc_vel, self.theta_exc_vel

    
    def get_rad_shell_mean(self):
        #print("get_rad_shell_mean")
        """
        Create array with radii equal to the mean of the radii of the particles in each shell 
        to get an array with one radius for each shell. This is necessary to perform the curve fit.
        """
        self.check_rad_shell()        
        self.check_splits()
        self.check_rad_pcls()
        
        # Make array for storing mean of distance from halo center to particles in each shell and cone
        self.rad_shell_mean = np.zeros([self.n_phi, self.n_theta, self.n_shell])
        
        for i, arrays in enumerate(self.idxs_arr_angles):
            for j, idxs_ang in enumerate(arrays):
                for k, idxs_rad in enumerate(self.idxs_arr_rad):
                    idxs_equal = np.nonzero(np.in1d(idxs_ang, idxs_rad))
                    idxs = idxs_ang[idxs_equal]

                    # Skips shells with no particles. 
                    # The radius of these shells are set to the mean of the upper and lower radius of the corresponding shell
                    if len(idxs) > 0:
                        self.rad_shell_mean[i][j][k] = np.mean(self.rad_pcls[idxs])
                    else:
                        self.rad_shell_mean[i][j][k] = (self.rad_shell[k] + self.rad_shell[k+1]) / 2
                    
        return self.rad_shell_mean
    
    
    def delete_zero_shells(self, rad_arr, vel_arr):
        """
        Delete shells with no particles for easier to exclude directions with velocity profiles
        which do not have the expected shape.
        """
        idxs_zeros = np.where(vel_arr == 0)[0] # Get indices of shells with no particles
        if len(idxs_zeros) > 0:
            return np.delete(rad_arr, idxs_zeros), np.delete(vel_arr, idxs_zeros), idxs_zeros
        else:
            return rad_arr, vel_arr, idxs_zeros

        
    def v_of_r(self, r, a, b):
        """ Power-law fitted to the velocity profiles in each cone. """
        return -a * (self.rvir / r)**b
    

    def fit_vel_profile(self, i, j):
        """
        Fit curve to the peculiar velocity profiles in the radial interval from rad_in to rad_out.
        """
        
        rad_list = self.rad_fit_list
        
        params_list = []
        idxs_list = []
        v_fit_list = []
        
        for k in range(len(rad_list)-1):
            # Find indices of shells in the radial interval chosen for the fit 
            idxs = np.where((self.rad_shell_mean[i][j] > rad_list[k]*self.rvir) & (self.rad_shell_mean[i][j] < rad_list[k+1]*self.rvir))[0]
            rad_arr = self.rad_shell_mean[i][j][idxs]
            vel_arr = self.v_pec_cones[i][j][idxs]
    
            # Delete shells with no particles
            rad_arr_new, vel_arr_new, idxs_zeros = self.delete_zero_shells(rad_arr, vel_arr)
            idxs_arr_new = np.delete(np.array(idxs), idxs_zeros)

            # Only perform fit if length of velocity array with no zero shells is longer than 10% of the the original array
            if len(vel_arr_new) > 0.1*len(vel_arr) and len(vel_arr_new) > 1:
                params, cov = curve_fit(self.v_of_r, rad_arr_new, vel_arr_new, p0=[0.1,0.1], maxfev=100000)
                params_list.append(params)
                v_fit = self.v_of_r(rad_arr_new, params[0], params[1])
                v_fit_list.append(v_fit)
                idxs_list.append(idxs_arr_new)
            else:
                params_list.append(np.array([None, None])) # Set elements to None if the velocity array is too short for a fit to be performed
    
        return np.array(params_list, dtype=np.ndarray), idxs_list, np.array(v_fit_list, dtype=np.ndarray)
    
        
    def get_vel_diffs(self, v_fits):
        """Compute the relative difference between the velocities at the end and start of adjacent fits."""
        diff_vel_list = []
        for k in range(len(v_fits)-1):
            diff_vel_list.append(abs(v_fits[k][-1] - v_fits[k+1][0]) / abs(v_fits[k][-1]))
        return np.array(diff_vel_list, dtype=np.ndarray)

    
    def neg_coeff_dirs(self, plot=False, path_out=None):
        #print("neg_coeff_dirs")
        """
        Fit function to velocity profiles. Outputs arrays with phi and theta of directions 
        where either of the power-law coefficients are negative.
        """
        
        assert hasattr(self, "vel_pcls") == True, "read_hdf5_file has to be run first to define the halo and particle properties!"
        
        if plot==True:
            assert path_out != None, "A path to the directory in which to store the figures must be given as input to neg_coeff_dirs!"
            print(f"Create and save plots of fits to the velocity profiles for each direction for Halo {self.n_halo}!")
            
        self.check_velocities_cones()
        self.check_rad_shell_mean()
        self.check_pos_vel_dirs()
        
        # Lists for storing indices, phi and theta of the cones corresponding to the directions which should be removed
        idxs_exc = []
        phi_exc = []
        theta_exc = []            
        
        neg_count = 0 # Count the number of directions with negative power-law coefficients / bad velocity profiles
        
        # Fit power-law to the velocity profile for each direction and remove the directions with negative coefficients
        n_dir = 0
        for i, j in np.ndindex(self.n_phi, self.n_theta):
            idxs = np.array([i, j])

            rad_arr = self.rad_shell_mean[i][j]
            rad_arr_all, vel_arr_all, idxs_zeros = self.delete_zero_shells(self.rad_shell_mean[i][j], self.v_pec_cones[i][j])
            
            if len(self.idxs_exc_vel) > 0:
                nequal = len(np.where(np.all(idxs == self.idxs_exc_vel, axis=1))[0])
            else:
                nequal = 0
                
            # Skip the directions which have already been excluded due to positive velocities
            if nequal > 0:
                #print("Skip, direction already excluded due to positive velocities!")
                continue
             
            params_all, idxs_all, v_fit_arr = self.fit_vel_profile(i, j)
            
            n_none = len(np.where(params_all == None)[0])
            
            # Skip direction if any of the intervals contain no particles
            if n_none > 0:
            #print("Skip, all shells in one of the intervals are empty!")
                continue
            
            # Get the difference between the end and start velocity of the adjacent fits
            diff_vels_arr = self.get_vel_diffs(v_fit_arr)

            # Check if any of the coefficients are negative 
            # and if any of the ends and starts of the intervals are larger than a chosen threshold
            n_neg_coeffs = len(np.where(params_all < 0)[0])
            
            if self.use_fitting_threshold == True:
                n_above_threshold = len(np.where(diff_vels_arr > self.fitting_profile_threshold)[0])
            else:
                n_above_threshold = 0
                
            # Skip directions where the start and end of the fits in the different intervals are larger than a given offset
            if (n_neg_coeffs > 0) or (n_above_threshold > 0):
                neg_count += 1
                # Add the indices, phi and theta of the cones which should be excluded to lists if
                # any of the best-fit power-law coefficients are negative
                phi_exc.append([self.phi_cone[i], self.phi_cone[i+1]])
                theta_exc.append([self.theta_cone[j], self.theta_cone[j+1]])
                idxs_exc.append(idxs)
                xlabel = "excluded"
            else:
                xlabel = "included"

            n_dir += 1 # Count directions
            
            # Plot velocity curves and best fits for all directions
            if plot==True:
                plt.rcParams['figure.max_open_warning'] = 100 # Increase the number of figures that can be made before matplotlib raises warning of too many open figures         

                fig, ax = plt.subplots(figsize=(10,6))
                ax.set_xlabel(r"Radius $[\mathrm{R}_{\mathrm{vir}}]$ ({%s})" % xlabel)
                #ax.set_xlabel(xlabel)
                ax.set_ylabel("Peculiar velocity [km/s]")
                ax.plot(rad_arr_all/self.rvir, vel_arr_all)
                # Plot fits
                for k in range(len(v_fit_arr)):
                    x = rad_arr[idxs_all[k]]/self.rvir
                    y = v_fit_arr[k]
                    ax.plot(rad_arr[idxs_all[k]]/self.rvir, v_fit_arr[k])
                
                # Create figures with arrows pointing at the end and start of fits to illustrate how one of the exclusion principles work
#                    if k == 1:
#                        wn = 10
#                        # Add text and line connecting text to point
#                        ax.plot(x[0], y[0], "C2o")
#                        ax.plot(x[-1], y[-1], "C2o")
#                        plt.annotate(r"$v_\mathrm{fit,end,i}$", xy=(x[-1]-0.08, y[-1]+20), xytext=(x[-1]-3, y[-1]+400),
#                                     fontsize=22, arrowprops=dict(facecolor="black", width=0.5, headwidth=wn, headlength=wn, shrink=4.7))
#                    elif k == 2:
#                        ax.plot(x[0], y[0], "C3o")
#                        plt.annotate(r"$v_\mathrm{fit,start,i+1}$", xy=(x[0]+0.05, y[0]+20), xytext=(x[0]+1, y[0]+500),
#                                     fontsize=22, arrowprops=dict(facecolor="black", width=0.5, headwidth=wn, headlength=wn, shrink=4.7))
#                    else:
#                        ax.plot(x[-1], y[-1], "C1o")
                
                path_out_fits = os.path.join(path_out, f"velocity_fits_halo_{self.n_halo}_{self.cosmo_theory}_z={self.redshift}_dir_{n_dir}_{xlabel}.pdf")
                plt.savefig(path_out_fits, bbox_inches='tight')    
                
        #print("Directions where one of the power-law coefficients are negative: ", neg_count)

        self.idxs_exc_coeff = np.array(idxs_exc)
        self.phi_exc_coeff = np.array(phi_exc)
        self.theta_exc_coeff = np.array(theta_exc)

        return self.idxs_exc_coeff, self.phi_exc_coeff, self.theta_exc_coeff
        

    def get_rta(self, a, b):
        #print("get_rta")
        """
        Computes the turnaround radius by solving v_radial = v_peculiar + v_Hubble = 0 for the radius r. 
        """
        return (a * self.rvir**b / self.Hubble)**(1 / (1 + b))
        
    
    def get_idxs_to_fit(self):
        """
        Find indices of the excluded directions.
        """
        # Make new array with indices of both the directions with positive velocities 
        # and with negative power-law coefficients or only with one of them if there are no
        # directions containing the other
        if (len(self.idxs_exc_vel) == 0):
            idxs_exc = self.idxs_exc_coeff
        elif (len(self.idxs_exc_coeff) == 0):
            idxs_exc = self.idxs_exc_vel
        else:
            idxs_exc = np.concatenate((self.idxs_exc_vel, self.idxs_exc_coeff))

        print(f"Excluded directions: {len(idxs_exc)}")

        # Create array with only indices of directions which are included
        idxs_in_set = set(np.ndindex((self.n_phi, self.n_theta)))  # Indices of all directions
        idxs_exc = set(map(tuple, idxs_exc))                       # Indices of directions to be excluded
        idxs_to_fit = list(idxs_in_set - idxs_exc)                 # Indices of directions to be kept
        
        return idxs_to_fit
    
    
    def rta_of_phi_theta(self, plot=False, plot_fit=True, path_out=None):
        #print("rta_of_phi_theta")
        """
        Find the turnaround radius as function of the angles phi and theta.
        
        Input:
            - rad_in/rad_out: Inner and outer radius of the region considered when
                              fitting the velocity profiles to a power-law.
        Output:
            - Array with turnaround radii for the directions not excluded.
            - Array with turnaround radii for all directions. The turnaround radius is set
              to zero for excluded directions.
        """

        assert hasattr(self, "pos_pcls") == True, "read_hdf5_file has to be run first to define the halo and particle properties!"

        if plot==True:
            assert path_out != None, "A path to the directory in which to store the figures must be given as input to rta_of_phi_theta!"
            print(f"Create and save plots of fits to the velocity profiles for each direction for Halo {self.n_halo}!")
        
        # Get peculiar velocities of the particles to find the directions which should be removed
        self.get_vel_cones()

        # Get mean radii for each shell and find indices where the mean radii of the shells lie between 3 and 7 virial radii
        self.get_rad_shell_mean()

        if self.exclude_directions == True:
            # Get indices of the directions to exclude
            self.pos_vel_dirs()
            self.neg_coeff_dirs()
            idxs_to_fit = self.get_idxs_to_fit()
        else:
            self.pos_vel_dirs()
            self.idxs_exc_coeff = []
            idxs_to_fit = self.get_idxs_to_fit()
            
        rta_arr = np.zeros(len(idxs_to_fit)) # For storing the turnaround radii for the different directions as 1D array 
        rta_cones = np.zeros([self.n_phi, self.n_theta], dtype=float) # For storing the turnaround radii for the different directions as 2D array
        
        n_zeros = 0 # Count the number of directions skipped because of too many empty shells
        
        n_dir = 0 # Count directions
        
        # Fit curve to velocity profiles and use coefficients to compute turnaround radius
        # Find only the turnaround radius for the directions not excluded. The turnaround radii 
        # for the removed directions is set to 0 so that it does not contribute to the analysis
        for i, idxs in enumerate(idxs_to_fit):
            j, k = idxs[0], idxs[1]
            # Get indices of radii in chosen radial interval
            idxs_in_range = np.where((self.rad_shell_mean[j][k] > self.rad_in * self.rvir) & 
                                     (self.rad_shell_mean[j][k] < self.rad_out * self.rvir))
            vel_data = self.v_pec_cones[j][k][idxs_in_range]     # Pick velocities of particles in radial interval
            rad_data = self.rad_shell_mean[j][k][idxs_in_range]  # Pick radii of particles in radial interval
                        
            # Find shells with velocity zero meaning these shells contain no particles
            idxs_zero_shells = np.where(vel_data == 0)[0]

            # Either skip direction if all shells in an interval are empty,
            # or skip direction if more than 10% of the shells have no particles ???
            if (len(vel_data) == 0) or (len(idxs_zero_shells) > 0.1*len(vel_data)):
                #print("Skip direction because fit can not be performed when there are too many empty shells!") 
                n_zeros += 1
                continue
            
            # Skip directions with no particles between rad_in-rad_out virial radii
            if np.mean(vel_data) != 0:
                params, cov = curve_fit(self.v_of_r, rad_data, vel_data)  # Perform fit to velocity profiles for all directions
                
#                if np.isnan(params[0])==False:
                rta_val = self.get_rta(params[0], params[1])              # Compute the turnaround radius from the best-fit parameters
                rta_arr[i] = rta_val                                      # Add the turnaround radius to the 1D array
                rta_cones[j][k] = rta_val                                 # Add the turnaround radius to the 2D array (as function of angles phi and theta)
            
            n_dir += 1
            
            # Plot velocity curves and best fits for all directions
            if plot==True:
                plt.rcParams['figure.max_open_warning'] = 100 # Increase the number of figures that can be made before matplotlib raises warning of too many open figures         
                rad_arr_all, vel_arr_all, idxs_zeroz = self.delete_zero_shells(self.rad_shell_mean[j][k], self.v_pec_cones[j][k])
                fig, ax = plt.subplots(figsize=(10,6))
                ax.set_xlabel(r"Radius $[\mathrm{R}_{\mathrm{vir}}]$")
                ax.set_ylabel("Peculiar velocity [km/s]")
                ax.plot(rad_arr_all/self.rvir, vel_arr_all)
                if plot_fit==True:
                    ax.plot(rad_data/self.rvir, self.v_of_r(rad_data, params[0], params[1]))#, label="a = {:.2f}, b = {:.2f}".format(params[0], params[1]))
                    #plt.legend()
                    fit_str = "_wfit"
                else:
                    fit_str = "_nofit"
                path_out_fits = os.path.join(path_out, f"velocity_profile{fit_str}_full_halo_{self.n_halo}_{self.cosmo_theory}_z={self.redshift}_dir_{n_dir}.pdf")
                plt.savefig(path_out_fits, bbox_inches='tight')
        
        if n_zeros > 0:
            print(f"{n_zeros} directions were skipped because of too many empty shells!") if n_zeros > 1 else print(f"{n_zeros} direction were skipped because of too many empty shells!")
                
        # Delete directions with turnaround radius set to zero, which means these directions have no or very few particles
        idxs_zero_dirs = np.where(rta_arr == 0)[0]
        if len(idxs_zero_dirs) > 0:
            rta_arr = np.delete(rta_arr, idxs_zero_dirs)
        
        return rta_arr, rta_cones
    
    
    def rta_of_mass(self, halo_first, halo_last, lower_perc, upper_perc, path_halo_file):
        #print("rta_of_mass")
        """
        Find the turnaround radius as function of virial mass for a given number of halos.
        This function assumes the input files with halo and particle properties have the same
        names as they were given when the files were created with CreateHaloFile. 
        
        Input:
            - halo_first / halo_last: Halo number for the first and last halo wanted for analysis.
                                Takes values between 1 and maximum number of halos created with CreateHaloFile.
                                1 is the largest halo, 2 is the second largest and so on.
            - perc_lower / perc_upper: Upper and lower percentiles for error analysis.
            - fit_polynomial: If True, a polynomial fit is performed.
            - path_halo_file: Full path to directory where files containing halo and particle info created with 
                              CreateHaloFile are stored. If not given, the code assumes the directory 
                              is at the same location as the file running the code.
        
        Saves the following quantities of each halo to file:
            - Haloids.
            - Virial masses.
            - Mean, 50th, upper and lower percentiles of the turnaround radius.
            The number of halos is (n_last - n_first), but it could be shorter since
            for halos with few particles, all directions might be excluded and no 
            turnaround radius will be found.
        """
#       	print(f"Redshift: z = {self.redshift}") 
        assert halo_first < halo_last, "halo_last has to be larger than halo_first!"
        
#        if path_halo_file == None:
#            input_dir = "output_halo_info"
#        else:
#            input_dir = path_halo_info

        if self.exclude_directions == False:
            print("Exclude directions with positive velocities outside 2 virial radii!")
        elif self.exclude_directions == True and self.use_fitting_threshold == False:
            print("Exclude directions with positive velocities outside 2 virial radii and negative power-law coefficients!")
        else:
            print("Exclude directions with positive velocities outside 2 virial radii, negative power-law coefficients and velocity profiles with \"bad behaviour\"!")
    
        n_halos = halo_last + 1 - halo_first
        haloid, mvir, m200b, rta_mean_vir, rta_p50_vir, error_l_vir, error_u_vir, rta_mean_200b, rta_p50_200b, error_l_200b, error_u_200b = [[] for _ in range(11)]
        
        no_rta_list = []
        for i in range(n_halos):
            i_halo = i + halo_first
            print("Halo ", i_halo)
            file_path = f"{path_halo_file}/halo_{i_halo}_z={self.redshift}.h5"
            self.read_halo_file(file_path)

            rta_arr = self.rta_of_phi_theta()[0]
            print("Directions kept: ", len(rta_arr))
            if len(rta_arr) == 0:
                no_rta_list.append(str(i_halo))
            else:
                haloid.append(i_halo)
                mvir.append(self.mvir)
                m200b.append(self.m200b)
                rta_mean_vir.append(np.nanmean(rta_arr) / self.rvir)
                rta_mean_200b.append(np.nanmean(rta_arr) / self.r200b)
                
                # Find and save percentiles. When plotting, the mean should be subtracted from the error bars
                perc = np.percentile(rta_arr, (50, lower_perc, upper_perc))
                rta_p50_vir.append(perc[0] / self.rvir)
                error_l_vir.append(perc[0] / self.rvir)
                error_u_vir.append(perc[2] / self.rvir)
                rta_p50_200b.append(perc[0] / self.r200b)
                error_l_200b.append(perc[0] / self.r200b)
                error_u_200b.append(perc[2] / self.r200b)
                
        if len(no_rta_list) > 0:
            (print("No turnaround radius found for Halo " + ", ".join(no_rta_list) 
                   + " because all directions were excluded for this halo.") 
             if len(no_rta_list) == 1 else 
             print("No turnaround radius found for Halo " + ", ".join(no_rta_list) 
                   + " because all directions were excluded for these halos."))

        print("Number of halos: ", len(mvir))
        
        if self.exclude_directions == False:
            filename_end = "exclude_dirs_1_condition"
        elif self.exclude_directions == True and self.use_fitting_threshold == False:
            filename_end = "exclude_dirs_2_conditions"
        else:
            filename_end = "exclude_dirs_3_conditions"
        
        path_out = os.path.join(self.path_out, f"rta_of_mass_halo_{halo_first}_{halo_last}_{self.cosmo_theory}_{filename_end}_z={self.redshift}_new.h5")
               
        hf = h5.File(path_out, "w")
        hf.create_dataset("haloid", data=np.array(haloid))
        hf.create_dataset("mvir", data=np.array(mvir))
        hf.create_dataset("m200b", data=np.array(m200b))
        hf.create_dataset("rta_mean_vir", data=np.array(rta_mean_vir))
        hf.create_dataset("rta_p50_vir", data=np.array(rta_p50_vir))
        hf.create_dataset("error_lower_vir", data=np.array(error_l_vir))
        hf.create_dataset("error_upper_vir", data=np.array(error_u_vir))
        hf.create_dataset("rta_mean_200b", data=np.array(rta_mean_200b))
        hf.create_dataset("rta_p50_200b", data=np.array(rta_p50_200b))
        hf.create_dataset("error_lower_200b", data=np.array(error_l_200b))
        hf.create_dataset("error_upper_200b", data=np.array(error_u_200b))
        hf.close()


    def save_vel_profiles(self):
        """
        Save velocity profiles to file.
        Input:
            - haloid
            
        The output file contains the velocity profiles for each direction 
        """
        
#        assert hasattr(self, "pos_pcls") == True, "read_hdf5_file has to be run first to define the halo and particle properties!"
        
        print(f"Save file with radii and velocities of velocity profiles of Halo {self.n_halo}!")
    
        self.check_velocities_cones()
        self.check_excluded_directions()
        self.check_rad_shell_mean()
        
        v_pec = self.v_pec_cones
        v_tot = self.v_tot_cones
        pos_vel_dirs = self.idxs_exc_vel
        neg_coeff_dirs = self.idxs_exc_coeff
        rad_shell = self.rad_shell_mean    
        
        v_pec_list = []  # List for storing arrays with peculiar velocities for each direction
        v_tot_list = []  # List for storing arrays total velocities for each direction
        val_list = []     # List for storing integers telling if the direction was included or not 
        rad_list = []     # List for storing arrays with radii for each direction
        
        for i, j in np.ndindex(self.n_phi, self.n_theta):
            idx = np.array([i, j])
            
            if len(pos_vel_dirs) > 0:
                vel_list = np.where(np.all(idx == pos_vel_dirs, axis=1))[0]
            else:
                vel_list = []
        
            if len(neg_coeff_dirs) > 0:
                coeff_list = np.where(np.all(idx == neg_coeff_dirs, axis=1))[0]
            else:
                coeff_list = []
                
            if len(vel_list) > 0:
                val = 1
            elif len(coeff_list) > 0:
                val = 2        
            else:
                val = 0

            v_pec_list.append(v_pec[i][j])
            v_tot_list.append(v_tot[i][j])
            val_list.append(val)
            rad_list.append(self.rad_shell_mean[i][j])
        
        if self.exclude_directions == True:
            filename_end = "_dirs_excluded"
        else:
            filename_end = ""
        
        path_out = os.path.join(self.path_out, f"vel_profiles_halo_{self.n_halo}_{self.cosmo_theory}_z={self.redshift}{filename_end}.h5")
        
        hf = h5.File(path_out, "w")
        hf.create_dataset("v_pec", data=v_pec_list)
        hf.create_dataset("v_tot", data=v_tot_list)
        hf.create_dataset("rad", data=rad_list)
        hf.create_dataset("exc_vals", data=val_list)
        hf.close()
        
        
    def mass_resolution(self):
        """ Returns the mass resolution at the given reedshift. """
        a = 1 / (1 + self.redshift)
        L = self.boxsize * a
        mass = 0.8e11 * L**3 / self.n_grids
        return mass
    
    def save_density_profile(self):
        """
        Compute the density profile for each direction and save to file.
        """
        self.check_velocities_cones()
        self.check_rad_shell()
        self.check_angles_cones()
        
        assert hasattr(self, "boxsize") == True, "parse_parameter_file has to be run to define the boxsize and number of grids!"
        
        mass = self.mass_resolution()
        mass_arr = self.num_count * mass
        density_list = []
        
        delta_phi = self.phi_cone[1] - self.phi_cone[0]
        delta_r = self.rad_shell[1] - self.rad_shell[0]
        da = delta_phi * delta_r
        
        for i in range(self.n_phi):
            for j in range(self.n_theta-1):
                delta_theta = abs(self.theta_cone[j] - self.theta_cone[j+1])
                delta_volum = delta_theta * da
                density_list.append(mass_arr[i][j] / delta_volum)
                    
        if self.exclude_directions == True:
            filename_end = "_dirs_excluded"
        else:
            filename_end = ""
        
        path_out = os.path.join(self.path_out, f"density_halo_{self.n_halo}_{self.cosmo_theory}_z={self.redshift}{filename_end}.h5")
        
        hf = h5.File(path_out, "w")
        hf.create_dataset("rho", data=density_list)
        hf.close()