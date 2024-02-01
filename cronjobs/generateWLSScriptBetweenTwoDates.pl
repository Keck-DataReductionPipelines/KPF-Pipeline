#! /usr/local/bin/perl

use strict;
use warnings;

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


my $scriptfile = "runWLSPipelineFrom" . $yyyymmdd[0] . "To" . $yyyymmdd[$#yyyymmdd] . ".sh";

if (! open(SCR, ">$scriptfile") ) {
  die "*** Error: Could not open $scriptfile for writing; quitting...\n";
}

my $shebang = '#! /bin/bash -l';

print SCR "$shebang\n";

foreach my $yyyymmdd (@yyyymmdd) {
    print "yyyymmdd=$yyyymmdd\n";

    my @op = `cat runDailyPipelines.sh`;

    foreach my $op (@op) {
        if ($op =~ /^\s+$/) { next; }
        if ($op =~ /^procdate/) { next; }
        if ($op =~ /\/bin\/bash/) { next; }
        if ($op =~ /^printenv/) { next; }
        if ($op =~ /kpfmastersruncmd/) { next; }
        $op =~ s/\$procdate/$yyyymmdd/g;
        print SCR "$op";
    }
}

if (! close(SCR) ) {
    die "*** Error: Could not close $scriptfile; quitting...\n";
}

`chmod +x $scriptfile`;

exit 0;


