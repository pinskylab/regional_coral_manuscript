# regional_coral_manuscript

This repository contains simulated data and associated code for the manuscript 'Evolution and connectivity influence the persistence and recovery of coral reefs under climate change in the Caribbean, Southwest Pacific, and Coral Triangle.'

**Authors:**   
Lisa C. McManus<sup>1,2*</sup>, Daniel L. Forrest<sup>1*</sup>, Edward W. Tekwa<sup>1</sup>, Daniel E. Schindler<sup>3</sup>, Madhavi A. Colton<sup>4</sup>, Michael M. Webster<sup>5</sup>, Timothy E. Essington<sup>3</sup>, Stephen R. Palumbi<sup>6</sup>, Peter J. Mumby<sup>7</sup> and Malin L. Pinsky<sup>1</sup>

\* Lisa C. McManus and Daniel L. Forrest should be considered joint first author.

**Affiliations:**  
<sup>1</sup>Department of Ecology, Evolution, and Natural Resources, Rutgers University, New Brunswick, NJ, USA. </br> 
<sup>2</sup>Hawaiʻi Institute of Marine Biology, University of Hawaiʻi at Manoa, Kaneʻohe, HI 96744, USA. </br>
<sup>3</sup>School of Aquatic and Fishery Sciences, University of Washington, Seattle, WA, USA.  </br>
<sup>4</sup>Coral Reef Alliance, Oakland, CA, USA.  </br>
<sup>5</sup>Department of Environmental Studies, New York University, 285 Mercer St., New York, NY 10003, USA. </br>
<sup>6</sup>Department of Biology, Hopkins Marine Station, Stanford University, Pacific Grove, CA, USA. </br>
<sup>7</sup>Marine Spatial Ecology Laboratory, School of Biological Sciences, The University of Queensland, St Lucia, Queensland, Australia.  

**I. Simulations**  
We have included the Python scripts used for calculating numerical solutions on a high-performance computing system. Note that you may need to modify the syntax and install packages depending on the particular HPC that you are using. These data were generated on the <a href='https://oarc.rutgers.edu/resources/amarel/'>Rutgers Amarel cluster</a>.

Scripts for each region are in their respective directories: 'Caribbean,' 'CoralTriangle' and 'SouthwestPacific.' The following text will describe files in /Caribbean but will apply for the analogous files in /CoralTriangle and /Southwest Pacific.

1. functions_deterministic.py: contains all functions called in the numerical solver
2. parameters_caribbean.py: sets or loads all parameters
3. routine_caribbean.py: the actual routine to run simulations (produces data)
4. submit_caribbean.sh: file to submit 'routine_caribbean.py' to the HPC
5. \input directory: includes sea surface temperature, lats and lons, connectivity matrix

**II. Statistical Analyses**  
Simulation outputs (.npy) were converted to .csv for statistical analyses in R. 

**III. Figures**  
To recreate the figures in the main text and supporting information, use the Jupyter notebooks 'Main_Figures.ipynb' and 'Supplemental_Figures.ipynb,' respectively.
