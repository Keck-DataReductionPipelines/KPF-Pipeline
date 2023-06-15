#! /usr/local/bin/perl

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
use File::Path qw/make_path/;

select STDERR; $| = 1; select STDOUT; $| = 1;


# Log start time.

my ($startscript, $endscript, @checkpoint, $icheckpoint);

$startscript = time();
print "Start time = ", scalar localtime($startscript), "\n";
print "====================================================\n";
$icheckpoint = 0;
$checkpoint[$icheckpoint++] = $startscript;


# Read KPF-related environment variables.

# Legacy KPF directory for outputs of testing.
# E.g., /KPF-Pipeline-TestData
my $testdatadir = $ENV{KPFPIPE_TEST_DATA};

if (! (defined $testdatadir)) {
    die "*** Env. var. KPFPIPE_TEST_DATA not set; quitting...\n";
}

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


# Initialize fixed parameters and read command-line parameter.

my $iam = 'kpfmasters_wls_auto.pl';
my $version = '1.2';

my $procdate = shift @ARGV;                  # YYYYMMDD command-line parameter.

if (! (defined $procdate)) {
    die "*** Error: Missing command-line parameter YYYYMMDD; quitting...\n";
}

my $dockercmdscript = 'kpfmasters_wls_auto.sh';    # Auto-generates this shell script with multiple commands.
my $containerimage = 'kpf-drp:latest';
my $recipe = '/code/KPF-Pipeline/recipes/wls_auto.recipe';
my $config = '/code/KPF-Pipeline/configs/wls_auto.cfg';
my $sbxdir = "${sandbox}/masters/$procdate";


# Print environment.

print "iam=$iam\n";
print "version=$version\n";
print "procdate=$procdate\n";
print "dockercmdscript=$dockercmdscript\n";
print "containerimage=$containerimage\n";
print "recipe=$recipe\n";
print "config=$config\n";
print "KPFPIPE_TEST_DATA=$testdatadir\n";
print "KPFPIPE_MASTERS_BASE_DIR=$mastersdir\n";
print "KPFCRONJOB_SBX=$sandbox\n";
print "KPFCRONJOB_LOGS=$logdir\n";
print "KPFCRONJOB_CODE=$codedir\n";
print "Docker container name = $containername\n";


# Change directory to where the Dockerfile is located.

chdir "$codedir" or die "Couldn't cd to $codedir : $!\n";

my $script = "#! /bin/bash\n" .
             "make init\n" .
             "export PYTHONUNBUFFERED=1\n" .
             "git config --global --add safe.directory /code/KPF-Pipeline\n" .
             "kpf -r $recipe  -c $config --date ${procdate}\n" .
             "exit\n";
my $makescriptcmd = "echo \"$script\" > $dockercmdscript";
`$makescriptcmd`;
`chmod +x $dockercmdscript`;

`mkdir -p $sbxdir`;
`cp -pr ${mastersdir}/${procdate}/kpf_${procdate}*L1.fits $sbxdir`;

my $dockerruncmd = "docker run -d --name $containername " .
                   "-v ${codedir}:/code/KPF-Pipeline -v ${testdatadir}:/testdata -v $sandbox:/data " .
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


# Directory to store products should already exist because
# cronjob kpfmastersruncmd_l0.pl ran before.

my $destdir  = "${mastersdir}/$procdate";

if (! (-e $destdir)) {
    print "*** Error: Product directory does not exist ($destdir): $!\n";
    exit(64);
}

sleep(30);

my $globfiles = "$sbxdir/*master_WLS*";

my @files  = glob("$globfiles");

foreach my $file (@files) {
    my $destfile = "$destdir/$file";
    if (! (-e $destfile)) {
        if (! (copy($file, $destdir))) {
            die "*** Warning: couldn't copy $file to $destdir ($!); " .
                "quitting...\n";
        } else {
            print "Copied $file to $destdir\n";
        }
    }
}


# Log end time.

$endscript = time();
print "====================================================\n";
print "End time = ", scalar localtime($endscript), "\n";
print "Elapsed total time (sec.) = ", $endscript - $startscript, "\n";

print "Terminating normally...\n";


# Move log file from runtime directory to product directory.

my ($logfileBase) = $iam =~ /(.+)\.pl/;

my $logfile = $logdir . '/' . $logfileBase . '_' . $procdate . '.out';

if (-e $logfile) {

    if (! (move($logfile, $destdir))) {
        die "*** Warning: couldn't move $logfile to $destdir ($!); " .
            "quitting...\n";
    } else {
        print "Moved $logfile to $destdir\n";
    }
}


exit(0);
