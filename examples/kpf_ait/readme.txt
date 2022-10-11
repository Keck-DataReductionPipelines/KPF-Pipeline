Current implemntation:

In the recipe (kpf.recipe) and config (kpf.cfg) file,
do_order_trace, do_spectral_extraction, do_rv, and do_rv_reweighting are defined to detemine which pipeline module is to process.
overwrite is defined to determine if the data will be reproduced in case the data already exist.

for order trace (L0->csv & L0, non watch mode)
kpf -r examples/kpf_ait/kpf.recipe -c examples/kpf_ait/kpf_ot.cfg --date 20220517
(kpf_ot.cfg: set do_order_trace = True and other do_ = False)

for spectral extraction + RV + RV reweighting (L0->L1->L2 -> reweighted L2, watch mode)
kpf -r examples/kpf_ait/kpf.recipe -c examples/kpf_ait/kpf.cfg --watch /data/2D/20220517/
(kpf.recipe: set do_spectral_extraction = True, do_rv = True, do_rv_reweighting = True)

for RV + RV reweighting (L1->L2 -> reweighted L2, watch mode)
kpf -r examples/kpf_ait/kpf.recipe -c examples/kpf_ait/kpf.cfg --watch /data/L1/20220517/
(kpf.recipe: set do_rv = True & do_rv_reweighting, and all other do_xxx = False)

for spectral extraction + RV + RV reweighting (L0->L1->L2, non watch mode)
kpf -r examples/kpf_ait/kpf.recipe -c examples/kpf_ait/kpf.cfg --date /data/2D/20220517/

for RV + Rv reweighting (L1->L2 ->reweighted L2, non watch mode)
kpf -r examples/kpf_ait/kpf.recipe -c examples/kpf_ait/kpf.cfg --date /data/L1/20220517/

for RV reweighting only: (L2 -> weighted L2, non watch mode)
method 1. kpf -r examples/kpf_ait/kpf.recipe -c examples/kpf_ait/kpf.cfg --date /data/L2/20220517
(kpf.recipe: set do_rv_reweighting = True and other do_xxxx = False)
method 2: kpf -r examples/kpf_ait/kpf.recipe -c examples/kpf_ait/kpf_reweight.cfg --date 20220517
(kpf_reweight.cfg: set do_rv_reweighting = True and other do_xxxx = False)

for doing certain process(es) on some specific date, non watch mode
kpf -r examples/kpf_ait/kpf.recipe -c examples/kpf_ait/kpf.cfg --date 20220517
(set do_order_trace, do_spectral_extraction, do_rv, and do_rv_reweighting in kpf.cfg to be True or False as needed)


Outdated implementation:
default_kpf_new_data.cfg (for order trace and spectral extraction only) - outdated

output_dir = <the path point to folder containing L1>                            ex. /data/kpf/
output_dir_flat = <the path point to the folder containing  order trace result>  ex. /data/kpf/

input_dir = <the path point to the folder with 2D data>                          ex. /data/kpf/2D/<date>/
input_dir_flat = <the path point to the folder with 2D data>                     ex. /data/kpf/2D/<date_for_flat>/

input_flat_file_pattern = <filename without .fits>                               ex. KP.20220510.04445.31_2D
input_lev0_file_prefix = <level 0 file pattern for spectral extraction>          ex. KP.202205*


output_trace = <sub-directory under output_dir_flat with order trace result>     ex. adhoc/<date>/
output_exraction = <sub-directory under outout_dir with L1 result>               ex. L1/<date>/

output_rv = <sub-directory under output_dir with L2 result>                      ex. L2/<date>/

