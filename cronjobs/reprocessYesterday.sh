#!/bin/bash

export DATE=`/usr/bin/date -d "yesterday" +%Y%m%d`
/scr/doppler/production/KPF-Pipeline/scripts/kpf_slowtouch.sh -e -d /data/kpf/L0/${DATE}/