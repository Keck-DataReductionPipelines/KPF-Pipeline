#! /usr/local/bin/perl

##########################################################################
# Pipeline Perl script to do detached docker run.  Can run this script
# in the background so that open terminal is not required.
#
# Generate all KPF L0 master calibration files for YYYYMMDD date, given as
# command-line input parameter.  Input KPF L0 FITS files are copied to
# sandbox directory KPFPIPE_DATA=/data/user/rlaher/sbx, and outputs
# are written to KPFPIPE_TEST_DATA=/KPF-Pipeline-TestData, and then at
# the end copied to /data/kpf/masters/YYYYMMDD. The following two files
# must exist in directory KPFPIPE_TEST_DATA=/KPF-Pipeline-TestData:
# channel_orientation_ref_path_red = kpfsim_ccd_orient_red_2amp.txt
# channel_orientation_ref_path_green = kpfsim_ccd_orient_green.txt
# KPF 2D FITS files with no master bias subtraction are made as
# intermediate products in the sandbox.  The master-bias subtraction,
# master-dark subtraction, and master-flattening, as appropriate, are
# done by the individidual Frameworks for generation of the master bias,
# master dark, master flat, and various master arclamps.
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

my $iam = 'kpfmastersruncmd_l0.pl';
my $version = '1.0';

my $procdate = shift @ARGV;                  # YYYYMMDD command-line parameter.

if (! (defined $procdate)) {
    die "*** Error: Missing command-line parameter YYYYMMDD; quitting...\n";
}

my $dockercmdscript = 'kpfmasterscmd_l0.sh';    # Auto-generates this shell script with multiple commands.
my $containername = 'russkpfmastersdrpl0';
my $containerimage = 'kpf-drp:latest';
my $recipe = '/code/KPF-Pipeline/recipes/kpf_masters_drp.recipe';
my $config = '/code/KPF-Pipeline/configs/kpf_masters_drp.cfg';
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

`mkdir -p $sandbox/L0/$procdate`;
`mkdir -p $sandbox/2D/$procdate`;
`cp -pr /data/kpf/L0/$procdate/*.fits $sandbox/L0/$procdate`;

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


# Make directory to store products.

my $destdir  = "${mastersdir}/$procdate";

if (! (-e $destdir)) {
    if (! make_path($destdir)) {
        die "*** Error: Could not make directory ($destdir): $!\n";
    } else {
        print "Made new directory $destdir\n";
    }
}

sleep(30);

my $globfiles = "${testdatadir}/kpf_${procdate}*";

my @files  = glob("$globfiles");

foreach my $file (@files) {
    if (! (copy($file, $destdir))) {
        die "*** Warning: couldn't copy $file to $destdir ($!); " .
            "quitting...\n";
    } else {
        print "Copied $file to $destdir\n";
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
