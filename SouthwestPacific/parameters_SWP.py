#! Parameters for Coral Adaptation model
import numpy as np

# Choose the number of iterations
iterations = 1 #number of iterations per scenario

#! Load the temperature files
#! SWP

SST_SWP_45 = np.load("./SWP/input/SWP_SST_45_yearly.npy")
SST_SWP_85 = np.load("./SWP/input/SWP_SST_85_yearly.npy")

#! FIJI
#SST_Fiji_45_obs = np.load("./Fiji/input/Fiji_SST_45_yearly_obs.npy")
#SST_Fiji_85_obs = np.load("./Fiji/input/Fiji_SST_85_yearly_obs.npy")
#SST_Fiji_45_sim = np.load("./Fiji/input/Fiji_SST_45_yearly_sim.npy")
#SST_Fiji_85_sim = np.load("./Fiji/input/Fiji_SST_85_yearly_sim.npy")

#! CARIBBEAN
#SST_Caribbean_45_obs = np.load("./Caribbean/input/Caribbean_SST_45_yearly_obs.npy")
#SST_Caribbean_85_obs = np.load("./Caribbean/input/Caribbean_SST_85_yearly_obs.npy")
#SST_Caribbean_45_sim = np.load("./Caribbean/input/Caribbean_SST_45_yearly_sim.npy")
#SST_Caribbean_85_sim = np.load("./Caribbean/input/Caribbean_SST_85_yearly_sim.npy")

#! INDONESIA
#SST_Indonesia_45_obs = np.load("./Indonesia/input/Indonesia_SST_45_yearly_obs.npy")
#SST_Indonesia_85_obs = np.load("./Indonesia/input/Indonesia_SST_85_yearly_obs.npy")
#SST_Indonesia_45_sim = np.load("./Indonesia/input/Indonesia_SST_45_yearly_sim.npy")
#SST_Indonesia_85_sim = np.load("./Indonesia/input/Indonesia_SST_85_yearly_sim.npy")

#! Load the connectivity matrices:
D_SWP = np.load("./SWP/input/D_SWP_2001_revised.npy")
#D_Fiji = np.load("./Fiji/input/D_1997_Fiji.npy")
#D_Fiji = np.load("./data/D_1999_Fiji.npy")
#D_Fiji = np.load("./Fiji/input/D_2001_Fiji.npy")
#D_Caribbean = np.load("./Caribbean/input/D_Caribbean.npy")
#D_Indonesia = np.load("./Indonesia/input/D_Indonesia.npy")

#! Load the area files:
areas_SWP = np.load("./SWP/input/SWP_reef_area.npy")
#areas_Caribbean = np.load("./Caribbean/input/Caribbean_reef_areas.npy")
#areas_Indonesia = 

#! Load the subregion file:
subregion_ID = np.load("./SWP/input/subregion_SWP_Fiji.npy")

#! Load the betweenness centrality file:
bw_centrality_sorted = np.load("./SWP/input/SWP_bet_central_sorted.npy")
egvec_centrality_sorted = np.load("./SWP/input/SWP_egvec_centrality_sorted.npy")
even_space = np.load("./SWP/input/SWP_even_space.npy")

#! Other parameters
nsp = 2 # Number of species in model
species_type = np.array([[1,1]]) # Species type ID
species = ["C1","C2"] # Species labels
r_max = np.array([[1.5,1.5]])
w = np.array([[1.,3.]])
alphas = np.array([[5.77,0.9],[0.9,5.77]]) 
mortality_model = "temp_vary"
m_const = 0.1

#! Change these values
V = np.array([[0.0,0.0]]) # V = 0., 0.01, 0.1
beta = np.array([[0.0,0.0]]) #beta = 0, 0.05, 0.5
region = "SWP"
SST_45 = SST_SWP_45
SST_85 = SST_SWP_85
hindcast_label = "H"
temp_scenario1 = "45" 
temp_scenario2 = "85" 
D = D_SWP.T
hindcast_length = 149
areas = areas_SWP
algmort_min = 1.0
algmort_max = 1.0
# Strategy options: none, hot, cold, hotcold, space, highcoral, lowcoral, random, egvec_cent, bw_cent, even_space
reserve_strategy = "none" 
reserve_fraction = 0.0
# Change this to True if setting the MPA strategy only for the subregion
subregion_flag = False