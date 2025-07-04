#! /usr/bin/perl

##########################################################################
# Pipeline Perl script to do detached docker run.  Can run this script
# in the background so that open terminal is not required.
#
# Run standalone recipe to generate the order-trace files for GREEN and RED.
# Input is master flat for the YYYYMMDD date, which is given as a
# command-line input parameter.  Input KPF 2D master-flat calibration file is
# copied to sandbox directory /data/user/rlaher/sbx/masters/YYYYMMDD, and
# outputs are written to the same directory, and then finally copied to
# /data/kpf/masters/YYYYMMDD.
#
# Also generates a daily order-mask file with GREEN and RED FITS extensions.
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
my $containername = $ENV{KPFCRONJOB_DOCKER_NAME_ORDERTRACE};

if (! (defined $containername)) {
    die "*** Env. var. KPFCRONJOB_DOCKER_NAME_ORDERTRACE not set; quitting...\n";
}

my $trunctime = time() - int(53 * 365.25 * 24 * 3600);   # Subtract off number of seconds in 53 years (since 00:00:00 on January 1, 1970, UTC).
$containername .= '_' . $$ . '_' . $trunctime;           # Augment container name with unique numbers (process ID and truncated seconds).


# Database user for connecting to the database to run this script and query L0Files database table.
# E.g., kpfporuss
my $dbuser = $ENV{KPFDBUSER};

if (! (defined $dbuser)) {
    die "*** Env. var. KPFDBUSER not set; quitting...\n";
}

# Database name of KPF operations database containing the L0Files table.
# E.g., kpfopsdb
my $dbname = $ENV{KPFDBNAME};

if (! (defined $dbname)) {
    die "*** Env. var. KPFDBNAME not set; quitting...\n";
}


# Initialize fixed parameters and read command-line parameter.

my $iam = 'kpfmasters_order_trace.pl';
my $version = '1.0';

my $procdate = shift @ARGV;                  # YYYYMMDD command-line parameter.

if (! (defined $procdate)) {
    die "*** Error: Missing command-line parameter YYYYMMDD; quitting...\n";
}

my $dockercmdscript = 'jobs/kpfmasters_order_trace';               # Auto-generates this shell script with multiple commands.
$dockercmdscript .= '_' . $$ . '_' . $trunctime . '.sh';           # Augment with unique numbers (process ID and truncated seconds).
my $containerimage = 'russkpfmasters:latest';
my $recipe = '/code/KPF-Pipeline/recipes/create_order_trace_files.recipe';
my $config = '/code/KPF-Pipeline/configs/create_order_trace_files.cfg';
my $recipe2 = '/code/KPF-Pipeline/recipes/create_order_rectification_file.recipe';
my $config2 = '/code/KPF-Pipeline/configs/create_order_rectification_file.cfg';
my $recipe3 = '/code/KPF-Pipeline/recipes/create_order_mask_file.recipe';
my $config3 = '/code/KPF-Pipeline/configs/create_order_mask_file.cfg';
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
print "recipe2=$recipe2\n";
print "config2=$config2\n";
print "recipe3=$recipe3\n";
print "config3=$config3\n";
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
             "mkdir -p /data/masters/${procdate}\n" .
             "cp -p /masters/${procdate}/kpf_${procdate}_master_flat.fits /data/masters/${procdate}\n" .
             "kpf -r $recipe  -c $config --date ${procdate}\n" .
             "cp -p /data/masters/${procdate}/*.csv /masters/${procdate}\n" .
             "kpf -r $recipe2  -c $config2 --date ${procdate}\n" .
             "cp -p /data/masters/${procdate}/kpf_${procdate}_master_flat_*.fits /masters/${procdate}\n" .
             "kpf -r $recipe3  -c $config3 --date ${procdate}\n" .
             "cp -p /data/masters/${procdate}/*order_mask.fits /masters/${procdate}\n" .
             "cp -p /data/logs/${procdate}/pipeline_${procdate}.log /masters/${procdate}/pipeline_order_trace_${procdate}.log\n" .
             "exit\n";
my $makescriptcmd = "echo \"$script\" > $dockercmdscript";
`$makescriptcmd`;
`chmod +x $dockercmdscript`;

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
