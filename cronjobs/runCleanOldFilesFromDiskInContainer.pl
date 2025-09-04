#! /usr/bin/perl

##########################################################################
# Pipeline Perl script to do detached docker run.  Can run this script
# in the background so that open terminal is not required.
#
# Cleans up masters-pipeline sandbox (e.g., /data/user/rlaher/sbx)
# by executing cleanOldFilesFromDisk.pl inside a container for
# root delete privileges.
##########################################################################

use strict;
use warnings;
use File::Copy;

select STDERR; $| = 1; select STDOUT; $| = 1;


# Log start time.

my ($startscript, $endscript, @checkpoint, $icheckpoint);

$startscript = time();
print "Start time = ", scalar localtime($startscript), "\n";
print "====================================================\n";
$icheckpoint = 0;
$checkpoint[$icheckpoint++] = $startscript;


# Read required KPF-related environment variables.

# Sandbox directory for intermediate files.
# E.g., /data/user/rlaher/sbx
my $sandbox = $ENV{KPFCRONJOB_SBX};

if (! (defined $sandbox)) {
    die "*** Env. var. KPFCRONJOB_SBX not set; quitting...\n";
}

# Code directory of KPF-Pipeline git repo where the docker run command is executed.
# E.g., /data/user/rlaher/git/KPF-Pipeline
my $codedir = $ENV{KPFCRONJOB_CODE};

if (! (defined $codedir)) {
    die "*** Env. var. KPFCRONJOB_CODE not set; quitting...\n";
}

# Logs directory where log file (STDOUT) from this script goes (see runDailyPipelines.sh).
# Normally this is the code directory of KPF-Pipeline git repo.
# E.g., /data/user/rlaher/git/KPF-Pipeline
my $logdir = $ENV{KPFCRONJOB_LOGS};

if (! (defined $logdir)) {
    die "*** Env. var. KPFCRONJOB_LOGS not set; quitting...\n";
}

# Docker container name for this Perl script, a known name so it can be monitored by docker ps command.
# E.g., russcleanupsbx
my $containername = $ENV{KPFCRONJOB_DOCKER_NAME_CLEANUP_SBX};

if (! (defined $containername)) {
    die "*** Env. var. KPFCRONJOB_DOCKER_NAME_CLEANUP_SBX not set; quitting...\n";
}

my $trunctime = time() - int(53 * 365.25 * 24 * 3600);   # Subtract off number of seconds in 53 years (since 00:00:00 on January 1, 1970, UTC).
$containername .= '_' . $$ . '_' . $trunctime;           # Augment container name with unique numbers (process ID and truncated seconds).

# Docker container image name for this Perl script.
# E.g., russkpfmasters:latest
my $containerimage = $ENV{KPFCRONJOB_DOCKER_IMAGE_NAME};

if (! (defined $containerimage)) {
    die "*** Env. var. KPFCRONJOB_DOCKER_IMAGE_NAME not set; quitting...\n";
}

# Check if Docker image exists
my $image_check = `docker images -q $containerimage 2>/dev/null`;
chomp $image_check;
if (!$image_check) {
    print "*** Error: Docker image '$containerimage' not found!\n";
    print "*** To build this image, run the following command from the KPF-Pipeline root directory:\n";
    print "***   docker build -t $containerimage .\n";
    print "*** Or if you want to use the existing kpf-drp image, set:\n";
    print "***   export KPFCRONJOB_DOCKER_IMAGE_NAME=kpf-drp:latest\n";
    die "*** Quitting due to missing Docker image...\n";
} else {
    print "*** Docker image '$containerimage' found (ID: $image_check)\n";
}


# Initialize fixed parameters and read command-line parameter.

my $iam = 'runCleanOldFilesFromDiskInContainer.pl';
my $version = '1.0';

my $procdate = shift @ARGV;                  # YYYYMMDD command-line parameter.

if (! (defined $procdate)) {
    die "*** Error: Missing command-line parameter YYYYMMDD; quitting...\n";
}

if (! ($procdate =~ /^\d\d\d\d\d\d\d\d$/)) {
    die "*** Error: Command-line parameter YYYYMMDD contains extra characters or digits; quitting...\n";
}

# These parameters are fixed for this Perl script.
my $dockercmdscript = 'jobs/runCleanOldFilesFromDiskInContainer';                     # Auto-generates this shell script with multiple commands.
$dockercmdscript .= '_' . $$ . '_' . $trunctime . '.sh';           # Augment with unique numbers (process ID and truncated seconds).
my $insidecontainersandbox = '/sbx';
my $insidecontainercode = '/code/KPF-Pipeline';


# Print environment.

print "iam=$iam\n";
print "version=$version\n";
print "procdate=$procdate\n";
print "dockercmdscript=$dockercmdscript\n";
print "containerimage=$containerimage\n";
print "KPFCRONJOB_SBX=$sandbox\n";
print "KPFCRONJOB_LOGS=$logdir\n";
print "KPFCRONJOB_CODE=$codedir\n";
print "KPFCRONJOB_DOCKER_NAME_CLEANUP_SBX=$containername\n";
print "Docker container name = $containername\n";
print "insidecontainersandbox = $insidecontainersandbox\n";
print "insidecontainercode = $insidecontainercode\n";


# Change directory to where the Dockerfile is located.

chdir "$codedir" or die "Couldn't cd to $codedir : $!\n";

my $script = "#! /bin/bash\n" .
             "export KPFCRONJOB_SBX=$insidecontainersandbox\n" .
             "${insidecontainercode}/cronjobs/cleanOldFilesFromDisk.pl >& ${insidecontainercode}/jobs/cleanOldFilesFromDisk_${procdate}.out\n" .
             "exit\n";
my $makescriptcmd = "echo \"$script\" > $dockercmdscript";
`$makescriptcmd`;
`chmod +x $dockercmdscript`;

my $dockerruncmd = "docker run -d --name $containername " .
                   "-v ${codedir}:$insidecontainercode -v $sandbox:$insidecontainersandbox " .
                   "$containerimage bash ./$dockercmdscript";
print "Executing $dockerruncmd\n";
my $opdockerruncmd = `$dockerruncmd`;
print "Output from dockerruncmd: $opdockerruncmd\n";


# Poll to see if done.

while (1) {
    my $cmd = "docker ps | grep $containername";
    my @op = `$cmd`;
    my $n = @op;
    if ($n == 0) { print "Exiting while loop...\n"; last; }
    for (my $i = 0; $i < @op; $i++) {
        my $op = $op[$i];
        chomp $op;
        print "i=$i: $op\n";
    }
    my $timestamp = localtime;
    print "[$timestamp] Sleeping 300 seconds...\n";
    sleep(300);
}

my $dockerrmcmd = "docker rm $containername";
print "Executing $dockerrmcmd\n";
my $opdockerrmcmd = `$dockerrmcmd`;
print "Output from dockerrmcmd: $opdockerrmcmd\n";


# Checkpoint

$checkpoint[$icheckpoint] = time();
printf "Elapsed time to run recipe (sec.) = %d\n",
       $checkpoint[$icheckpoint] - $checkpoint[$icheckpoint-1];
$icheckpoint++;


# Log end time.

$endscript = time();
print "====================================================\n";
print "End time = ", scalar localtime($endscript), "\n";
print "Elapsed total time (sec.) = ", $endscript - $startscript, "\n";

print "Terminating normally...\n";


exit(0);
