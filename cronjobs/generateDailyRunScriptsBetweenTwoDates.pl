#! /usr/local/bin/perl

use strict;
use warnings;

###########################################################################################
# This perl script requires the sqlite3 command to be installed:
# [rlaher@shrek ~]$ which sqlite3
# ~/sw/anaconda3/bin/sqlite3
#
# Instructions: Provide start and end dates on command line.  E.g.,
# export KPFCRONJOB_CODE=/data/user/rlaher/git/KPF-Pipeline
# mkdir -p ${KPFCRONJOB_CODE}/cronjobs/current_jobs
# cd ${KPFCRONJOB_CODE}/cronjobs/current_jobs
# ${KPFCRONJOB_CODE}/cronjobs/generateDailyRunScriptsBetweenTwoDates.pl 20230601 20230831
###########################################################################################

my $startyyyymmdd = shift @ARGV;                  # YYYYMMDD command-line parameter.
my $endyyyymmdd = shift @ARGV;                    # YYYYMMDD command-line parameter.

if ((! (defined $startyyyymmdd)) or (! ($startyyyymmdd =~ /^\d\d\d\d\d\d\d\d$/))) {
    die "startyyyymmdd either not defined or not correct format; quitting...\n";
}

if ((! (defined $endyyyymmdd)) or (! ($endyyyymmdd =~ /^\d\d\d\d\d\d\d\d$/))) {
    die "endyyyymmdd either not defined or not correct format; quitting...\n";
}

my ($year, $month, $day);

($year, $month, $day) = $startyyyymmdd =~ /(\d\d\d\d)(\d\d)(\d\d)/;
my $startdate = $year . '-' . $month . '-' . $day;

($year, $month, $day) = $endyyyymmdd =~ /(\d\d\d\d)(\d\d)(\d\d)/;
my $enddate = $year . '-' . $month . '-' . $day;


my $cmdforjdstart = "sqlite3 test.db \"SELECT julianday(\'$startdate 00:00:00.0\');\"";
print "Executing cmd = [$cmdforjdstart]\n";
my $computedjdstart = `$cmdforjdstart`;
chomp $computedjdstart;
print "computedjdstart = $computedjdstart\n";


my $cmdforjdend = "sqlite3 test.db \"SELECT julianday(\'$enddate 00:00:00.0\');\"";
print "Executing cmd = [$cmdforjdend]\n";
my $computedjdend = `$cmdforjdend`;
chomp $computedjdend;
print "computedjdend = $computedjdend\n";

my (@yyyymmdd);

for (my $i = int($computedjdstart); $i <= int($computedjdend); $i++) {

    my $jdi = $i + 0.5;
    my $cmd = "sqlite3 test.db \"SELECT datetime($jdi);\"";

    #print "Executing cmd = [$cmd]\n";

    my $computedatetime = `$cmd`;
    chomp $computedatetime;

    my ($obsyear, $obsmonth, $obsday) = $computedatetime =~ /(\d\d\d\d)-(\d\d)-(\d\d)/;

    my $obsdate = $obsyear . $obsmonth . $obsday;
    print "jdi, obsdate = $jdi, $obsdate\n";

    push @yyyymmdd, $obsdate;
}

my @reverse_yyyymmdd = reverse @yyyymmdd;


my $pwd = $ENV{"PWD"};
print "PWD=$pwd\n";

my $cronjob_code = $ENV{"KPFCRONJOB_CODE"};

my $scriptdir = $pwd;
my $scriptfile = "runMastersPipeline_From_${startyyyymmdd}_To_${endyyyymmdd}.sh";

if (! open(SCR, ">$scriptfile") ) {
  die "*** Error: Could not open $scriptfile for writing; quitting...\n";
}

my $shebang = '#! /bin/bash -l';

print SCR "$shebang\n";

foreach my $yyyymmdd (@reverse_yyyymmdd) {
    print "yyyymmdd=$yyyymmdd\n";

    my $outfile = "runDailyPipelines_${yyyymmdd}.sh";

    if (! open(OUT, ">$outfile") ) {
        die "*** Error: Could not open $outfile for writing; quitting...\n";
    }

    my @op = `cat $cronjob_code/cronjobs/runDailyPipelines.sh`;

    foreach my $op (@op) {
        if ($op=~/^procdate/) {next;}
        $op =~ s/\$procdate/$yyyymmdd/g;
        print OUT "$op";
    }

    if (! close(OUT) ) {
        die "*** Error: Couldn't close $outfile; quitting...\n";
    }

    `chmod +x $outfile`;

    my $cmd = $scriptdir . '/runDailyPipelines_' . 
              $yyyymmdd . 
              '.sh >& ' . $cronjob_code . '/runDailyPipelines_' .
	      $yyyymmdd .
              '.out &';

    print SCR "echo \"Executing command: $cmd\"\n";
    print SCR "$cmd\n";
    print SCR "echo \"Sleeping 2400 seconds\"\n";
    print SCR "sleep 2400\n";
}

if (! close(SCR) ) {
    die "*** Error: Could not close $scriptfile; quitting...\n";
}

`chmod +x $scriptfile`;

exit 0;

