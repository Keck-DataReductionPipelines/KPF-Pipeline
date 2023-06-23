#! /usr/local/bin/perl

##########################################################################
# Pipeline Perl script to do detached docker run.  Can run this script
# in the background so that an open terminal window is not required.
#
# Usage:
# ./l0files_register_in_db 202306?? (file glob)
#
# For the given date on the command line (or file glob that 
# covers a range of dates), ingest all L0 files in
# /data/kpf/L0/202306??/KP*fits on the shrek machine.
#
# This script will be run as a daily cronjob.
# It can also be executed in parallel for different inputs.
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

# Base directory of L0 files for permanent storage.
# E.g., /data/kpf/L0
my $l0dir = $ENV{KPFPIPE_L0_BASE_DIR};

if (! (defined $l0dir)) {
    die "*** Env. var. KPFPIPE_L0_BASE_DIR not set; quitting...\n";
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
my $templogdir = $ENV{KPFCRONJOB_LOGS};

if (! (defined $templogdir)) {
    die "*** Env. var. KPFCRONJOB_LOGS not set; quitting...\n";
}

# Docker container name for this Perl script, a known name so it can be monitored by docker ps command.
# E.g., russkpfl0registerindb
my $containername = 'russkpfl0registerindb';

my $trunctime = time() - int(53 * 365.25 * 24 * 3600);   # Subtract off number of seconds in 53 years (since 00:00:00 on January 1, 1970, UTC).
$containername .= '_' . $$ . '_' . $trunctime;           # Augment container name with unique numbers (process ID and truncated seconds).


# Database user for connecting to the database to run this script and insert records into the L0Files table.
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

my $iam = 'l0files_register_in_db.pl';
my $version = '1.0';
my $verbose = 1;

my $procdate = shift @ARGV;                  # YYYYMMDD command-line parameter.

if (! (defined $procdate)) {
    die "*** Error: Missing command-line parameter YYYYMMDD; quitting...\n";
}

my $dockercmdscript = 'jobs/l0registerdbcmd';                      # Auto-generates this shell script with multiple commands.
$dockercmdscript .= '_' . $$ . '_' . $trunctime . '.sh';           # Augment with unique numbers (process ID and truncated seconds).
my $containerimage = 'kpf-drp:latest';
my $recipe = '/code/KPF-Pipeline/recipes/quality_control_exposure.recipe';
my $inputconfig = $codedir . '/configs/quality_control_exposure.cfg';
my $config = 'quality_control_exposure';
$config .= '_' . $$ . '_' . $trunctime . '.cfg';                   # Augment with unique numbers (process ID and truncated seconds).

my ($basedir) = '/data/L0';
my $configtomodify = $codedir . '/jobs/' . $config;
&modifyConfigFile($basedir, $procdate, $inputconfig, $configtomodify, $iam, $verbose);


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

my $logdir = $l0dir;
$logdir =~ s/L0/logs/;
$logdir .= '/db';


# Print environment.

print "iam=$iam\n";
print "version=$version\n";
print "procdate=$procdate\n";
print "dockercmdscript=$dockercmdscript\n";
print "containerimage=$containerimage\n";
print "dbenvfilename=$dbenvfilename\n";
print "recipe=$recipe\n";
print "inputconfig=$inputconfig\n";
print "config=$config\n";
print "configtomodify=$configtomodify\n";
print "KPFPIPE_L0_BASE_DIR=$l0dir\n";
print "KPFCRONJOB_LOGS=$templogdir\n";
print "KPFCRONJOB_CODE=$codedir\n";
print "dbuser=$dbuser\n";
print "dbname=$dbname\n";
print "dbport=$dbport\n";
print "dbenvfile=$dbenvfile\n";
print "dbenvfileinside=$dbenvfileinside\n";
print "Docker container name = $containername\n";
print "basedir=$basedir\n";
print "logdir=$logdir\n";


# Change directory to where the Dockerfile is located.

chdir "$codedir" or die "Couldn't cd to $codedir : $!\n";

my $script = "#! /bin/bash\n" .
             "source $dbenvfileinside\n" .
             "make init\n" .
             "export PYTHONUNBUFFERED=1\n" .
             "pip install psycopg2-binary\n" .
             "git config --global --add safe.directory /code/KPF-Pipeline\n" .
             "kpf -r $recipe -c /code/KPF-Pipeline/jobs/$config\n" .
             "exit\n";
my $makescriptcmd = "echo \"$script\" > $dockercmdscript";
`$makescriptcmd`;
`chmod +x $dockercmdscript`;

my $dockerruncmd = "docker run -d --name $containername " .
                   "-v ${codedir}:/code/KPF-Pipeline -v ${l0dir}:${basedir} " .
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


# Move log file from runtime directory to product directory, assuming
# that the following convention for log-file naming is followed.

my ($logfileBase) = $iam =~ /(.+)\.pl/;

my $logfile = $templogdir . '/' . $logfileBase . '_' . $procdate . '.out';

if (-e $logfile) {

    if (! (move($logfile, $logdir))) {
        print "*** Warning: couldn't move $logfile to $logdir; " .
            "quitting...\n";
        exit(64);
    } else {
        print "Moved $logfile to $logdir\n";
    }
}


exit(0);


#------------------------------------------
# Subroutine to edit onfig file to change to the correct observation date.
#

sub modifyConfigFile {

    my ($basedir, $procdate, $inputconfigfile, $configfile, $iam, $verbose) = @_;

    print "inconfigfile=$inputconfigfile\n";
    print "configfile=$configfile\n";

    if (! (-e $inputconfigfile)) {
        print "*** Error: Input config file does not exist ($inputconfigfile); quitting...\n";
        exit(64);
    }


    # Edit the input config file.

    my $cmd = "cat $inputconfigfile";

    print "Executing command=[$cmd]\n";
    my (@lines) = `$cmd`;

    if (! open(OUT, ">$configfile") ) {
        print "*** Error: Couldn't open $configfile for writing; quitting...\n";
        exit(64);
    }

    foreach my $line (@lines) {

        chomp $line;

        #######################################################################
        # Change this line:
        #
        # lev0_fits_file_glob = /data/L0/20230620/KP.*.fits
        #######################################################################

        if ($line =~ /^lev0_fits_file_glob\s*=/) {
            $line = 'lev0_fits_file_glob = ' . $basedir . '/' . $procdate . '/KP.*.fits';
        }

        print OUT "$line\n";
    }

    if (! close(OUT) ) {
        print "*** Error: Couldn't close $configfile; quitting...\n";
        exit(64);
    }
}
