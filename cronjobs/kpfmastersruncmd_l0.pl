#! /usr/bin/perl

##########################################################################
# Pipeline Perl script to do detached docker run.  Can run this script
# in the background so that open terminal is not required.
#
# Generate all KPF L0 master calibration files for YYYYMMDD date, given as
# command-line input parameter.  Input KPF L0 FITS files are copied to
# sandbox directory KPFCRONJOB_SBX=/data/user/rlaher/sbx, and outputs
# are written to /data/user/rlaher/sbx/masters/pool, and then at
# the end copied to /data/kpf/masters/YYYYMMDD. The following two files
# must exist in /code/KPF-Pipeline/static:
# channel_orientation_ref_path_red = kpfsim_ccd_orient_red.txt
# channel_orientation_ref_path_green = kpfsim_ccd_orient_green.txt
# KPF 2D FITS files with no master bias subtraction are made as
# intermediate products in the sandbox.  The master-bias subtraction,
# master-dark subtraction, and master-flattening, as appropriate, are
# done by the individidual Frameworks for generation of the master bias,
# master dark, master flat, and various master arclamps.
#
# Be sure to make a jobs subdirectory for temporary files:
# mkdir -p $KPFCRONJOB_CODE/jobs
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
my $containername = $ENV{KPFCRONJOB_DOCKER_NAME_L0};

if (! (defined $containername)) {
    die "*** Env. var. KPFCRONJOB_DOCKER_NAME_L0 not set; quitting...\n";
}

my $trunctime = time() - int(53 * 365.25 * 24 * 3600);   # Subtract off number of seconds in 53 years (since 00:00:00 on January 1, 1970, UTC).
$containername .= '_' . $$ . '_' . $trunctime;           # Augment container name with unique numbers (process ID and truncated seconds).


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

my $iam = 'kpfmastersruncmd_l0.pl';
my $version = '2.4';

my $procdate = shift @ARGV;                  # YYYYMMDD command-line parameter.

if (! (defined $procdate)) {
    die "*** Error: Missing command-line parameter YYYYMMDD; quitting...\n";
}

if (! ($procdate =~ /^\d\d\d\d\d\d\d\d$/)) {
    die "*** Error: Command-line parameter YYYYMMDD contains extra characters or digits; quitting...\n";
}

# These parameters are fixed for this Perl script.
my $dockercmdscript = 'jobs/kpfmasterscmd_l0';                     # Auto-generates this shell script with multiple commands.
$dockercmdscript .= '_' . $$ . '_' . $trunctime . '.sh';           # Augment with unique numbers (process ID and truncated seconds).
my $containerimage = 'russkpfmasters:latest';
my $recipe = '/code/KPF-Pipeline/recipes/kpf_masters_drp.recipe';
my $config = '/code/KPF-Pipeline/configs/kpf_masters_drp.cfg';

my $configenvar = $ENV{KPFCRONJOB_CONFIG_L0};

if (defined $configenvar) {
    $config = $configenvar;
}

my $pythonscript = 'scripts/make_smooth_lamp_pattern_new.py';

my ($pylogfileDir, $pylogfileBase) = $pythonscript =~ /(.+)\/(.+)\.py/;
my $pylogfile = $pylogfileBase . '_' . $procdate . '.out';

my $pythonscript2 = 'scripts/reformat_smooth_lamp_fitsfile_for_kpf_drp.py';

my ($pylogfileDir2, $pylogfileBase2) = $pythonscript2 =~ /(.+)\/(.+)\.py/;
my $pylogfile2 = $pylogfileBase2 . '_' . $procdate . '.out';

my $pythonscript3 = 'database/scripts/cleanupMastersOnDiskAndDatabaseForDate.py';

my ($pylogfileDir3, $pylogfileBase3) = $pythonscript3 =~ /(.+)\/(.+)\.py/;
my $pylogfile3 = $pylogfileBase3 . '_' . $procdate . '.out';


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
print "pythonscript=$pythonscript\n";
print "pylogfile=$pylogfile\n";
print "pythonscript2=$pythonscript2\n";
print "pylogfile2=$pylogfile2\n";
print "pythonscript3=$pythonscript3\n";
print "pylogfile3=$pylogfile3\n";
print "KPFPIPE_MASTERS_BASE_DIR=$mastersdir\n";
print "KPFCRONJOB_SBX=$sandbox\n";
print "KPFCRONJOB_LOGS=$logdir\n";
print "KPFCRONJOB_CODE=$codedir\n";
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
             "rm -rf /data/masters/${procdate}\n" .
             "find /data/masters/pool/kpf_????????_master_*fits -mtime +7 -exec rm {} +\n" .
             "kpf -r $recipe  -c $config --date ${procdate}\n" .
             "python $pythonscript /data/masters/pool/kpf_${procdate}_master_flat.fits /data/masters/pool/kpf_${procdate}_smooth_lamp_orig.fits >& ${pylogfile}\n" .
             "python $pythonscript2 /data/masters/pool/kpf_${procdate}_smooth_lamp_orig.fits /data/masters/pool/kpf_${procdate}_master_flat.fits /data/masters/pool/kpf_${procdate}_smooth_lamp.fits >& ${pylogfile2}\n" .
             "rm /data/masters/pool/kpf_${procdate}_smooth_lamp_orig.fits\n" .
             "python $pythonscript3 $procdate >& ${pylogfile3}\n" .
             "mkdir -p /masters/${procdate}\n" .
             "sleep 3\n" .
             "cp -p /data/masters/pool/kpf_${procdate}* /masters/${procdate}\n" .
             "chown root:root /masters/${procdate}/*\n" .
             "cp -p /data/logs/${procdate}/pipeline_${procdate}.log /masters/${procdate}/pipeline_masters_drp_l0_${procdate}.log\n" .
             "cp -p /code/KPF-Pipeline/${pylogfile} /masters/${procdate}\n" .
             "cp -p /code/KPF-Pipeline/${pylogfile2} /masters/${procdate}\n" .
             "rm /code/KPF-Pipeline/${pylogfile}\n" .
             "rm /code/KPF-Pipeline/${pylogfile2}\n" .
             "exit\n";
my $makescriptcmd = "echo \"$script\" > $dockercmdscript";
`$makescriptcmd`;
`chmod +x $dockercmdscript`;

`mkdir -p $sandbox/L0/$procdate`;
`mkdir -p $sandbox/2D/$procdate`;
`cp -pr /data/kpf/L0/$procdate/*.fits $sandbox/L0/$procdate`;

my $dockerruncmd = "docker run -d --name $containername " .
                   "-v ${codedir}:/code/KPF-Pipeline -v $sandbox:/data -v ${mastersdir}:/masters " .
                   "--network=host -e DBPORT=$dbport -e DBNAME=$dbname -e DBUSER=$dbuser -e DBSERVER=127.0.0.1 " .
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
