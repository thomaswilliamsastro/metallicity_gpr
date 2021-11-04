This directory contains Gaussian Process Regression (GPR) smoothed metallicity maps (and associated
errors), v1p0. These maps use the pg16_scal metallicity calibration. The maps themselves are named as
'GALAXY_metallicity.fits', and the errors as 'GALAXY_metallicity_err.fits'. The units are 12+log(O/H). 
These maps have been generated using MUSE DR2.1, and sample table v1p6. There are also some diagnostic plots for each galaxy that show sigma-based residuals, and an overall overview of the map and error. There is also a summary plot for all galaxies that shows the model improvement (if any) of including the GPR. If there are any questions, please feel free to email me at williams@mpia.de.

CHANGES
v1.0: Accepted paper version (2021/11/04)v0.5: Moved to O'Donnell '94 extinction curve. Moved to restricting pixels within HII regions from the nebulae catalogues. (2021/07/22)
v0.4: Fixed double Galactic extinction correction. (2021/06/17)
v0.3: Updated to v1.6 of sample table. Updated to MUSE DR2.0. Uses common spatial maps for the GPR fitting. (2021/09/03)
v0.2: Updated to v1.5 of sample table (which means IC5332 is now included). Fixed important
deprojection error. Includes some diagnostic plots. (2020/07/27)
v0.1: Initial release (2020/06/24)
