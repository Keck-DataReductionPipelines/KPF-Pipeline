#!/bin/bash

export DATE=`/usr/bin/date -d "yesterday" +%Y%m%d`
/scr/doppler/production/KPF-Pipeline/scripts/kpf_slowtouch.sh -d /data/kpf/L0/${DATE}/

export DATE=`/usr/bin/date -d "2 days ago" +%Y%m%d`
/scr/doppler/production/KPF-Pipeline/scripts/kpf_slowtouch.sh -d /data/kpf/L0/${DATE}/
