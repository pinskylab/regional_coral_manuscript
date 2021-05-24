# regional_coral_manuscript

Citable as <a href="https://doi.org/10.5281/zenodo.4784134"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4784134.svg" alt="DOI"></a>.

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

Scripts for each region are in their respective directories: 'Caribbean,' 'CoralTriangle' and 'SouthwestPacific.' The following text will describe files in /Caribbean but will apply for the analogous files in /CoralTriangle and /Southwest Pacific. Note that output files were originally saved in an /output folder within each region directory but were subsequently moved to the main level /Output directory above.

1. functions_deterministic.py: contains all functions called in the numerical solver
2. parameters_caribbean.py: sets or loads all parameters
3. routine_caribbean.py: the main routine to run simulations (produces data)
4. submit_caribbean.sh: file to submit 'routine_caribbean.py' to the HPC
5. /input directory: includes sea surface temperature, lats and lons, connectivity matrix

**II. Statistical Analyses**  
Input files for this section are loaded from /Output.

1. /Python/Dataframes_for_models.ipynb: Construct Python dataframes and convert .npy to .csv for statistical analyses in R.  These are saved to /R/python_to_R_csv
2. /R/Regional_coral_stats.Rmd: Generates the stats model outputs (GLM) that are saved into /R/output/

**III. Figures**  
All figures were generated in Jupyter notebooks located in the /Python directory. To recreate the figures in the main text and supporting information, use 'Main_Figures.ipynb' and 'Supplemental_Figures.ipynb,' respectively. 
