#! /bin/bash

echo "Processing date: $(date +\%Y\%m\%d)"
/data/user/rlaher/git/KPF-Pipeline/cronjobs/kpfmastersruncmd_l0.pl $(date +\%Y\%m\%d) >& /data/user/rlaher/git/KPF-Pipeline/kpfmastersruncmd_l0_$(date +\%Y\%m\%d).out
echo Return value from kpfmastersruncmd_l0.pl = $?
/data/user/rlaher/git/KPF-Pipeline/cronjobs/kpfmastersruncmd_l1.pl $(date +\%Y\%m\%d) >& /data/user/rlaher/git/KPF-Pipeline/kpfmastersruncmd_l1_$(date +\%Y\%m\%d).out
echo Return value from kpfmastersruncmd_l1.pl = $?
