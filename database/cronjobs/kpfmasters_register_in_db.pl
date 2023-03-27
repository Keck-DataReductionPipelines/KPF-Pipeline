#! /usr/local/bin/perl

##########################################################################
# Pipeline Perl script to registerCalFilesForDate.py from within a
# docker container where python3 and the required packages are installed.
# Database credentials are read from the ~/.pgpass file.
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

# Legacy KPF port for Jupyter notebook.
# E.g., 6107
my $kpfpipeport = $ENV{KPFPIPE_PORT};

if (! (defined $kpfpipeport)) {
    die "*** Env. var. KPFPIPE_PORT not set; quitting...\n";
}

# Legacy KPF directory for outputs of testing (and Jupyter notebook).
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
my $containername = $ENV{KPFCRONJOB_DOCKER_NAME_DBSCRIPT};

if (! (defined $containername)) {
    die "*** Env. var. KPFCRONJOB_DOCKER_NAME_DBSCRIPT not set; quitting...\n";
}

# Database user for connecting to the database to run this script and insert records into the CalFiles table.
# E.g., kpfporuss
my $dbuser = $ENV{KPFDBUSER};

if (! (defined $dbuser)) {
    die "*** Env. var. KPFDBUSER not set; quitting...\n";
}


# Initialize fixed parameters and read command-line parameter.

my $iam = 'kpfmasters_register_in_db.pl';
my $version = '1.1';

my $procdate = shift @ARGV;                  # YYYYMMDD command-line parameter.

if (! (defined $procdate)) {
    die "*** Error: Missing command-line parameter YYYYMMDD; quitting...\n";
}

my $pythonscript = 'database/scripts/registerCalFilesForDate.py';
my $dockercmdscript = 'kpfmasters_register_in_db.sh';    # Auto-generates this shell script with multiple commands.
my $containerimage = 'kpf-drp:latest';

my ($pylogfileDir, $pylogfileBase) = $pythonscript =~ /(.+)\/(.+)\.py/;
my $pylogfile = $pylogfileBase . '_' . $procdate . '.out';

my $destdir  = "${mastersdir}/$procdate";

my $dbname = 'kpfopsdb';
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


# Print environment.

print "iam=$iam\n";
print "version=$version\n";
print "procdate=$procdate\n";
print "dockercmdscript=$dockercmdscript\n";
print "containerimage=$containerimage\n";
print "destdir=$destdir\n";
print "pythonscript=$pythonscript\n";
print "pylogfile=$pylogfile\n";
print "dbuser=$dbuser\n";
print "dbname=$dbname\n";
print "dbport=$dbport\n";
print "KPFPIPE_PORT=$kpfpipeport\n";
print "KPFPIPE_TEST_DATA=$testdatadir\n";
print "KPFPIPE_MASTERS_BASE_DIR=$mastersdir\n";
print "KPFCRONJOB_SBX=$sandbox\n";
print "KPFCRONJOB_LOGS=$logdir\n";
print "KPFCRONJOB_CODE=$codedir\n";
print "KPFCRONJOB_DOCKER_NAME_DBSCRIPT=$containername\n";


# Change directory to where the Dockerfile is located.

chdir "$codedir" or die "Couldn't cd to $codedir : $!\n";

my $script = "#! /bin/bash\nmake init\nexport PYTHONUNBUFFERED=1\npip install psycopg2-binary\npython $pythonscript $procdate >& ${pylogfile}\nexit\n";
my $makescriptcmd = "echo \"$script\" > $dockercmdscript";
`$makescriptcmd`;
`chmod +x $dockercmdscript`;

my $dockerrmcmd = "docker rm $containername";
print "Executing $dockerrmcmd\n";
my $opdockerrmcmd = `$dockerrmcmd`;
print "Output from dockerrmcmd: $opdockerrmcmd\n";

my $dockerruncmd = "docker run -d --name $containername -p 6207:6207 -e KPFPIPE_PORT=$kpfpipeport " .
                   "-v ${codedir}:/code/KPF-Pipeline -v ${mastersdir}:/masters " .
                   "--network=host -e DBPORT=$dbport -e DBNAME=$dbname -e DBUSER=$dbuser -e DBPASS=\"$dbpass\" -e DBSERVER=127.0.0.1 " .
                   "$containerimage bash ./$dockercmdscript";
#print "Executing $dockerruncmd\n";                                # COMMENT OUT THIS LINE: DO NOT PRINT DATABASE PASSWORD TO LOGFILE!
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
    print "[$timestamp] Sleeping 30 seconds...\n";
    sleep(30);
}


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


# Move the Python-script log file from runtime directory to product directory.

if (-e $pylogfile) {

    if (! (move($pylogfile, $destdir))) {
        die "*** Warning: couldn't move $pylogfile to $destdir ($!); " .
            "quitting...\n";
    } else {
        print "Moved $pylogfile to $destdir\n";
    }
}


# Move the log file generated by this Perl script to product directory, assuming
# that the following convention for log-file naming is followed.

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
