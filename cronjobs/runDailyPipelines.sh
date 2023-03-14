#! /bin/bash

echo "Processing date: $(date +\%Y\%m\%d)"
/data/user/rlaher/git/KPF-Pipeline/cronjobs/kpfmastersruncmd_l0.pl $(date +\%Y\%m\%d) >& /data/user/rlaher/git/KPF-Pipeline/kpfmastersruncmd_l0_$(date +\%Y\%m\%d).out
echo Return value from kpfmastersruncmd_l0.pl = $?
/data/user/rlaher/git/KPF-Pipeline/cronjobs/kpfmastersruncmd_l1.pl $(date +\%Y\%m\%d) >& /data/user/rlaher/git/KPF-Pipeline/kpfmastersruncmd_l1_$(date +\%Y\%m\%d).out
echo Return value from kpfmastersruncmd_l1.pl = $?
/data/user/rlaher/git/KPF-Pipeline/cronjobs/kpfmasters_wls_auto.pl $(date +\%Y\%m\%d) >& /data/user/rlaher/git/KPF-Pipeline/kpfmasters_wls_auto_$(date +\%Y\%m\%d).out
echo Return value from kpfmasters_wls_auto.pl = $?
/data/user/rlaher/git/KPF-Pipeline/database/cronjobs/kpfmasters_register_in_db.pl $(date +\%Y\%m\%d) >& /data/user/rlaher/git/KPF-Pipeline/kpfmasters_register_in_db_$(date +\%Y\%m\%d).out
echo Return value from kpfmasters_register_in_db.pl = $?
