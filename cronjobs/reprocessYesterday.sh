#!/bin/bash

export DATE=`/usr/bin/date -s 30 -d "yesterday" +%Y%m%d`
/scr/doppler/production/KPF-Pipeline/scripts/kpf_slowtouch.sh -d /data/kpf/L0/${DATE}/

export DATE=`/usr/bin/date -s 30 -d "2 days ago" +%Y%m%d`
/scr/doppler/production/KPF-Pipeline/scripts/kpf_slowtouch.sh -d /data/kpf/L0/${DATE}/
