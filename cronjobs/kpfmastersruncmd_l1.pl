#! /usr/local/bin/perl

##########################################################################
# Pipeline Perl script to do detached docker run.  Can run this script
# in the background so that open terminal is not required.
#
# Generate all KPF L1 master calibration files for YYYYMMDD date, given as
# command-line input parameter.  Input KPF L0 master calibration files are
# copied to sandbox directory /data/user/rlaher/sbx/masters/YYYYMMDD, and
# outputs are written to the same directory, and then finally copied to
# /data/kpf/masters/YYYYMMDD.  The following fixed files must exist in
# the /data/user/rlaher/sbx/masters/masters directory, as specified in
# config file kpf_masters_l1.cfg:
# wls_fits = ['masters/MasterLFCWLS.fits', 'masters/MasterLFCWLS.fits']
# hk_dark_fits = masters/KP.20221029.21537.28.fits
# hk_trace_path = masters/kpfMaster_HKOrderBounds20220909.csv
# hk_wavelength_path = ["masters/kpfMaster_HKwave20220909_sci.csv",
#                       "masters/kpfMaster_HKwave20220909_sky.csv"]
# masterbias_path = /data/masters/master_bias_20221022.fits
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


# Initialize parameters.

my $iam = 'kpfmastersruncmd_l1.pl';
my $version = '1.0';

my $procdate = shift @ARGV;                  # YYYYMMDD command-line parameter.

if (! (defined $procdate)) {
    die "*** Error: Missing command-line parameter YYYYMMDD; quitting...\n";
}

my $dockercmdscript = 'kpfmasterscmd_l1.sh';    # Auto-generates this shell script with multiple commands.
my $containername = 'russkpfmastersl1';
my $containerimage = 'kpf-drp:latest';
my $recipe = '/code/KPF-Pipeline/recipes/kpf_drp.recipe';
my $config = '/code/KPF-Pipeline/configs/kpf_masters_l1.cfg';
my $sandbox = '/data/user/rlaher/sbx';
my $codedir = '/data/user/rlaher/git/KPF-Pipeline';
my $testdatadir = '/KPF-Pipeline-TestData';
my $mastersdir = '/data/kpf/masters';
my $logdir = '/data/user/rlaher/git/KPF-Pipeline';


# Print environment.

print "iam=$iam\n";
print "version=$version\n";
print "procdate=$procdate\n";
print "dockercmdscript=$dockercmdscript\n";
print "containername=$containername\n";
print "containerimage=$containerimage\n";
print "recipe=$recipe\n";
print "config=$config\n";
print "sandbox=$sandbox\n";
print "codedir=$codedir\n";
print "testdatadir=$testdatadir\n";
print "mastersdir=$mastersdir\n";
print "logdir=$logdir\n";


# Change directory to where the Dockerfile is located.

chdir "$codedir" or die "Couldn't cd to $codedir : $!\n";

my $script = "#! /bin/bash\nmake init\nexport PYTHONUNBUFFERED=1\nkpf -r $recipe  -c $config --date ${procdate}\nexit\n";
my $makescriptcmd = "echo \"$script\" > $dockercmdscript";
`$makescriptcmd`;
`chmod +x $dockercmdscript`;

my $dockerrmcmd = "docker rm $containername";
print "Executing $dockerrmcmd\n";
my $opdockerrmcmd = `$dockerrmcmd`;
print "Output from dockerrmcmd: $opdockerrmcmd\n";

`mkdir -p $sandbox/masters/$procdate`;
`cp -pr ${mastersdir}/${procdate}/kpf_${procdate}*.fits ${sandbox}/masters/$procdate`;

my $dockerruncmd = "docker run -d --name $containername -p 6207:6207 -e KPFPIPE_PORT=6107 " .
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

my $globfiles = "${sandbox}/masters/$procdate/*";

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


# Move log file from runtime directory to product directory, assuming
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
