#!/bin/bash

# Script to update all Docker container launches with memory optimization
# This applies the memory optimization to all Perl scripts that launch Docker containers

set -e

echo "Updating all Docker container launches with memory optimization..."

# Find all Perl files that contain "docker run"
perl_files=$(grep -l "docker run" /home/bfulton/code/KPF-Pipeline/cronjobs/*.pl /home/bfulton/code/KPF-Pipeline/database/cronjobs/*.pl 2>/dev/null || true)

for file in $perl_files; do
    echo "Processing: $file"
    
    # Check if already updated (contains memory optimization)
    if grep -q "get_docker_memory_flags.sh" "$file"; then
        echo "  Already updated, skipping"
        continue
    fi
    
    # Create backup
    cp "$file" "$file.backup"
    
    # Find the docker run command and add memory optimization
    # This is a simplified approach - may need manual adjustment for complex cases
    sed -i 's/my \$dockerruncmd = "docker run -d --name \$containername "/my $memory_flags = `$codedir\/scripts\/get_docker_memory_flags.sh`;\nchomp($memory_flags);\n\nmy $dockerruncmd = "docker run -d --name $containername " .\n                   # Memory optimization\n                   "$memory_flags " .\n                   "/' "$file"
    
    echo "  Updated successfully"
done

echo "Update complete!"
echo "Note: Some files may need manual review for complex Docker commands."
echo "Backup files created with .backup extension."
