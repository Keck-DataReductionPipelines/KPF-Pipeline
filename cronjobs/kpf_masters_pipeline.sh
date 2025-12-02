#! /bin/bash -l

# Runs inside container.

if [ "${PROCDATE:+x}" ]; then
    echo "PROCDATE is set and not empty."
    procdate=$PROCDATE
    echo "Running the masters pipeline for $procdate"
    python /code/KPF-Pipeline/cronjobs/kpf_masters_pipeline.py $procdate >& /code/KPF-Pipeline/kpf_masters_pipeline_${procdate}.out
else
    echo "PROCDATE is not set or is empty; quitting..."
    exit 64
fi

exit
