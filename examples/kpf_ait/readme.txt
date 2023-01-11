Note: make update to 

1.  default_kpf_new_data.cfg (for order trace and spectral extraction only) 

output_dir = <the path point to folder containing L1>                            ex. /data/kpf/
output_dir_flat = <the path point to the folder containing  order trace result>  ex. /data/kpf/

input_dir = <the path point to the folder with 2D data> 			 ex. /data/kpf/2D/<date>/
input_dir_flat = <the path point to the folder with 2D data>			 ex. /data/kpf/2D/<date_for_flat>/

input_flat_file_pattern = <filename without .fits>                               ex. KP.20220510.04445.31_2D
input_lev0_file_prefix = <level 0 file pattern for spectral extraction> 	 ex. KP.202205*


output_trace = <sub-directory under output_dir_flat with order trace result>     ex. adhoc/<date>/
output_exraction = <sub-directory under outout_dir with L1 result> 		 ex. L1/<date>/

output_rv = <sub-directory under output_dir with L2 result> 			 ex. L2/<date>/



