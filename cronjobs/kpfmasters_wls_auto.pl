#! /usr/bin/perl

##########################################################################
# Pipeline Perl script to do detached docker run.  Can run this script
# in the background so that open terminal is not required.
#
# Run WLS on all KPF L1 master calibration files for YYYYMMDD date, given as
# command-line input parameter.  Input KPF L0 master calibration files are
# copied to sandbox directory /data/user/rlaher/sbx/masters/YYYYMMDD, and
# outputs are written to the same directory, and then finally copied to
# /data/kpf/masters/YYYYMMDD. The following fixed files must exist in
# the -v $sandbox:/data directory, as specified in
# config file wls_auto.cfg:
# master_wls_file = /data/L1_wls/MasterThArWLS_20230221.fits
# red_linelist = /data/masters/kpfMaster_ThArLines20221005_red.npy
# green_linelist = /data/masters/kpfMaster_ThArLines20230221_green.npy
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


# Read KPF-related environment variables.

# Base directory of master files for permanent storage.
# E.g., /data/kpf/masters
my $mastersdir = $ENV{KPFPIPE_MASTERS_BASE_DIR};

if (! (defined $mastersdir)) {
    die "*** Env. var. KPFPIPE_MASTERS_BASE_DIR not set; quitting...\n";
}

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
# E.g., russkpfmastersdrpl0
my $containername = $ENV{KPFCRONJOB_DOCKER_NAME_WLS};

if (! (defined $containername)) {
    die "*** Env. var. KPFCRONJOB_DOCKER_NAME_WLS not set; quitting...\n";
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


# Database user for connecting to the database to run this script and query CalFiles database table.
# E.g., kpfporuss
my $dbuser = $ENV{KPFDBUSER};

if (! (defined $dbuser)) {
    die "*** Env. var. KPFDBUSER not set; quitting...\n";
}

# Database name of KPF operations database containing the CalFiles table.
# E.g., kpfopsdb
my $dbname = $ENV{KPFDBNAME};

if (! (defined $dbname)) {
    die "*** Env. var. KPFDBNAME not set; quitting...\n";
}



# Initialize fixed parameters and read command-line parameter.

my $iam = 'kpfmasters_wls_auto.pl';
my $version = '2.0';

my $procdate = shift @ARGV;                  # YYYYMMDD command-line parameter.

if (! (defined $procdate)) {
    die "*** Error: Missing command-line parameter YYYYMMDD; quitting...\n";
}

my $dockercmdscript = 'jobs/kpfmasters_wls_auto';                  # Auto-generates this shell script with multiple commands.
$dockercmdscript .= '_' . $$ . '_' . $trunctime . '.sh';           # Augment with unique numbers (process ID and truncated seconds).

my $recipe = '/code/KPF-Pipeline/recipes/wls_auto.recipe';
my $config = '/code/KPF-Pipeline/configs/wls_auto.cfg';
my $sbxdir = "${sandbox}/masters/$procdate";


# Get database parameters from ~/.pgpass file.

my ($dbport, $dbpass);
my @op = `cat ~/.pgpass`;
foreach my $op (@op) {
    chomp $op;
    $op =~ s/^\s+|\s+$//g;  # strip blanks.
    if (($op =~ /$dbuser/) and ($op =~ /$dbname/)) {
        my (@f) = split(/\:/, $op);
        $dbport = $f[1];
        $dbpass = $f[4];
    }
}

my $dbenvfilename = "db";
$dbenvfilename .= '_' . $$ . '_' . $trunctime . '.env';                   # Augment with unique numbers (process ID and truncated seconds).
my $dbenvfile = "$codedir/jobs/" . $dbenvfilename;
my $dbenvfileinside = "/code/KPF-Pipeline/jobs/" . $dbenvfilename;

`touch $dbenvfile`;
`chmod 600 $dbenvfile`;
open(OUT,">$dbenvfile") or die "Could not open $dbenvfile ($!); quitting...\n";
print OUT "export DBPASS=\"$dbpass\"\n";
close(OUT) or die "Could not close $dbenvfile ($!); quitting...\n";



# Print environment.

print "iam=$iam\n";
print "version=$version\n";
print "procdate=$procdate\n";
print "dockercmdscript=$dockercmdscript\n";
print "containerimage=$containerimage\n";
print "recipe=$recipe\n";
print "config=$config\n";
print "KPFPIPE_MASTERS_BASE_DIR=$mastersdir\n";
print "KPFCRONJOB_SBX=$sandbox\n";
print "KPFCRONJOB_LOGS=$logdir\n";
print "KPFCRONJOB_CODE=$codedir\n";
print "Docker container name = $containername\n";
print "dbuser=$dbuser\n";
print "dbname=$dbname\n";
print "dbport=$dbport\n";
print "dbenvfile=$dbenvfile\n";
print "dbenvfileinside=$dbenvfileinside\n";
print "Docker container name = $containername\n";


# Change directory to where the Dockerfile is located.

chdir "$codedir" or die "Couldn't cd to $codedir : $!\n";

my $script = "#! /bin/bash\n" .
             "source $dbenvfileinside\n" .
             "make init\n" .
             "export PYTHONUNBUFFERED=1\n" .
             "git config --global --add safe.directory /code/KPF-Pipeline\n" .
             "mkdir -p /data/masters/${procdate}\n" .
             "cp -pr /masters/${procdate}/kpf_${procdate}*L1.fits /data/masters/${procdate}\n" .
             "kpf -r $recipe  -c $config --date ${procdate}\n" .
             "rm /masters/${procdate}/*master_WLS*\n" .
             "cp -p /data/masters/${procdate}/*master_WLS* /masters/${procdate}\n" .
             "mkdir -p /masters/${procdate}/wlpixelfiles\n" .
             "cp -p /data/masters/wlpixelfiles/*kpf_${procdate}* /masters/${procdate}/wlpixelfiles\n" .
             "cp -p /code/KPF-Pipeline/pipeline_${procdate}.log /masters/${procdate}/pipeline_wls_auto_${procdate}.log\n" .
             "rm /code/KPF-Pipeline/pipeline_${procdate}.log\n" .
             "exit\n";
my $makescriptcmd = "echo \"$script\" > $dockercmdscript";
`$makescriptcmd`;
`chmod +x $dockercmdscript`;

# Get memory optimization flags
my $memory_flags = `$codedir/scripts/get_docker_memory_flags.sh`;
chomp($memory_flags);

my $dockerruncmd = "docker run -d --name $containername " .
                   # Memory optimization
                   "$memory_flags " .
                   "-v ${codedir}:/code/KPF-Pipeline -v $sandbox:/data -v ${mastersdir}:/masters " .
                   "--network=host -e DBPORT=$dbport -e DBNAME=$dbname -e DBUSER=$dbuser -e DBSERVER=127.0.0.1 -e DBPASS=\"$dbpass\" " .
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
