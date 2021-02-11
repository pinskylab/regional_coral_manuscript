#! Parameters for Coral Adaptation model
import numpy as np

# Choose the number of iterations
iterations = 3 #number of iterations per scenario

#! Load the temperature files
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
SST_Indonesia_45_obs = np.load("./Indonesia/input/Indonesia_SST_45_yearly_obs_2083.npy")
SST_Indonesia_85_obs = np.load("./Indonesia/input/Indonesia_SST_85_yearly_obs_2083.npy")
#SST_Indonesia_45_sim = np.load("./Indonesia/input/Indonesia_SST_45_yearly_sim.npy")
#SST_Indonesia_85_sim = np.load("./Indonesia/input/Indonesia_SST_85_yearly_sim.npy")

#! Load the connectivity matrices:
#D_Fiji = np.load("./Fiji/input/D_1997_Fiji.npy")
#D_Fiji = np.load("./data/D_1999_Fiji.npy")
#D_Fiji = np.load("./Fiji/input/D_2001_Fiji.npy")
#D_Caribbean = np.load("./Caribbean/input/D_Caribbean.npy")
D_Indonesia = np.load("./Indonesia/input/D_Indo_2083_diag.npy")

#! Load the area files:
#areas_Fiji = np.load("./Fiji/input/Fiji_reef_areas.npy")
#areas_Caribbean = np.load("./Caribbean/input/Caribbean_reef_areas.npy")
areas_Indonesia = np.load("./Indonesia/input/Indonesia_reef_area_2083.npy")

#! Load the subregion file:
subregion_ID = np.load("./Indonesia/input/subregion_CT_LesserSunda.npy")

#! Load the betweenness centrality file:
bet_centrality_sorted = np.load("./Indonesia/input/Indonesia_bet_central_sorted.npy")

#! Other parameters
nsp = 3 # Number of species in model
species_type = np.array([[1,1,2]]) # Species type ID
species = ["C1","C2","M1"] # Species labels
r_max = np.array([[1.5,1.5,1.]])
w = np.array([[1,3,1.5]])
alphas = np.array([[1.,1.3,1.3],[1.,1.,1.3],[1.,1.,1.]])
mortality_model = "temp_vary"
m_const = 0.1

#! Change these values
V = np.array([[0.01,0.01,0.01]]) # V = 0., 0.01, 0.1
beta = np.array([[0.05,0.05,0.05]]) #beta = 0, 0.01, 0.1
region = "Indonesia"
SST_45 = SST_Indonesia_45_obs
SST_85 = SST_Indonesia_85_obs
hindcast_label = "H"
temp_scenario1 = "45"
temp_scenario2 = "85"
D = D_Indonesia.T
hindcast_length = 149
areas = areas_Indonesia
algmort_min = 0.1
algmort_max = 0.1
# Strategy options: none, hot, cold, hotcold, space, highcoral, lowcoral, random
reserve_strategy = "none"
reserve_fraction = 0.
# Change this to True if setting the MPA strategy only for the subregion
subregion_flag = False
