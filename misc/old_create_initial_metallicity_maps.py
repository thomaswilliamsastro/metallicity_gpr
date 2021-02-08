
if socket.gethostname() in ['astro-node4']:
    write_scale_lengths = True
else:
    write_scale_lengths = False

residual_ratios = []

scale_length_file_name = os.path.join(metallicity_dir, muse_version, 'scale_lengths')

if null_hypothesis:
    scale_length_file_name += '_null'
if use_radial_gpr and include_radial_subtract:
    scale_length_file_name += '_radial_gpr'
if use_pix_maps:
    scale_length_file_name += '_pix'

scale_length_file_name += '.fits'

scale_lengths = []
radial_ln_ls = []
gpr_ln_ls = []
