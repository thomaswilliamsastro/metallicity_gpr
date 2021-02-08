# The Two-Dimensional Metallicity Distribution of Nearby Galaxies

- [ ] Upgrade to DR2.0 copt maps
- [ ] Redo the metallicity maps at common resolution
- [ ] Swap convolving line maps out for convolved DAP products, and just regrid to the 150pc pixel size

This repository contains the code used in Williams et al. (2021 in prep.). The flow is as follows:

* Data Preparation:
  * `convolve_regrid_maps` convolves line maps to a common resolution of 150pc, and regrids to pixels of 150pc.
  * `pix_by_pix_metallicity_maps` will create metallicity maps from the convolved MUSE cubes in a pixel-by-pixel 
     manner
  * `metallicity_gpr` will TODO


