#! /usr/local/bin/perl

##########################################################################
# Pipeline Perl script to registerCalFilesForDate.py from within a
# docker container where python3 and the required packages are installed.
# Database credential are read from the ~/.pgpass file.
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

my $iam = 'kpfmasters_register_in_db.pl';
my $version = '1.0';

my $procdate = shift @ARGV;                  # YYYYMMDD command-line parameter.

if (! (defined $procdate)) {
    die "*** Error: Missing command-line parameter YYYYMMDD; quitting...\n";
}

my $codedir = '/data/user/rlaher/git/KPF-Pipeline';
my $pythonscript = 'database/scripts/registerCalFilesForDate.py';
my ($logfileDir, $logfileBase) = $pythonscript =~ /(.+)\/(.+)\.py/;
my $logfile = $logfileBase . '_' . $procdate . '.out';
my $dockercmdscript = 'kpfmasters_register_in_db.sh';    # Auto-generates this shell script with multiple commands.
my $containername = 'russkpfmastersregisterindb';
my $containerimage = 'kpf-drp:latest';
my $mastersdir = '/data/kpf/masters';
my $destdir  = "${mastersdir}/$procdate";

my $dbuser = 'kpfporuss';
my $dbname = 'kpfopsdb';
my ($dbport, $dbpass);
my @op = `cat /data/user/rlaher/.pgpass`;
foreach my $op (@op) {
    chomp $op;
    $op =~ s/^\s+|\s+$//g;  # strip blanks.
    if ($op =~ /$dbuser/) {
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
print "containername=$containername\n";
print "containerimage=$containerimage\n";
print "codedir=$codedir\n";
print "mastersdir=$mastersdir\n";
print "destdir=$destdir\n";
print "pythonscript=$pythonscript\n";
print "logfile=$logfile\n";
print "dbuser=$dbuser\n";
print "dbname=$dbname\n";
print "dbport=$dbport\n";


# Change directory to where the Dockerfile is located.

chdir "$codedir" or die "Couldn't cd to $codedir : $!\n";

my $script = "#! /bin/bash\nmake init\nexport PYTHONUNBUFFERED=1\npip install psycopg2-binary\npython $pythonscript $procdate >& ${logfile}\nexit\n";
my $makescriptcmd = "echo \"$script\" > $dockercmdscript";
`$makescriptcmd`;
`chmod +x $dockercmdscript`;

my $dockerrmcmd = "docker rm $containername";
print "Executing $dockerrmcmd\n";
my $opdockerrmcmd = `$dockerrmcmd`;
print "Output from dockerrmcmd: $opdockerrmcmd\n";

my $dockerruncmd = "docker run -d --name $containername -p 6207:6207 -e KPFPIPE_PORT=6107 " .
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


# Move log file from runtime directory to product directory, assuming
# that the following convention for log-file naming is followed.

if (-e $logfile) {

    if (! (move($logfile, $destdir))) {
        die "*** Warning: couldn't move $logfile to $destdir ($!); " .
            "quitting...\n";
    } else {
        print "Moved $logfile to $destdir\n";
    }
}


exit(0);
