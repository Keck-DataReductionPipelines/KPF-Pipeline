#! /usr/local/bin/perl

use strict;
use warnings;


my $yyyymm = shift @ARGV;                  # YYYYMM command-line parameter.

my (@yyyymmdd) = &getDaysInMonth($yyyymm);

my $scriptfile = "runMonthOfMasters_$yyyymm.sh";

if (! open(SCR, ">$scriptfile") ) {
  die "*** Error: Could not open $scriptfile for writing; quitting...\n";
}

my $shebang = '#! /bin/bash -l';

print SCR "$shebang\n";

foreach my $yyyymmdd (@yyyymmdd) {
    print "yyyymmdd=$yyyymmdd\n";

    my $outfile = "runDailyPipelines_${yyyymmdd}.sh";

    if (! open(OUT, ">$outfile") ) {
        die "*** Error: Could not open $outfile for writing; quitting...\n";
    }

    my @op = `cat runDailyPipelines.sh`;

    foreach my $op (@op) {
        if ($op=~/^procdate/) {next;}
        $op =~ s/\$procdate/$yyyymmdd/g;
        print OUT "$op";
    }

    if (! close(OUT) ) {
        die "*** Error: Couldn't close $outfile; quitting...\n";
    }

    `chmod +x $outfile`;

    my $cmd = '/data/user/rlaher/git/KPF-Pipeline/cronjobs/runDailyPipelines_' . 
              $yyyymmdd . 
              '.sh >& /data/user/rlaher/git/KPF-Pipeline/runDailyPipelines_' .
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



sub getDaysInMonth {

    my ($yymm) = @_;

    my ($year, $month) = $yymm =~ /(\d\d\d\d)(\d\d)/;

    my $day = 1;
    
    my ($zpDay) = &zeroPadTwoPlaces($day);

    my ($nightdate) = $year . '-' . $month . '-' . $zpDay;
    my $yyyymmdd = $year . $month . $zpDay;

    my $mm = $month;
    
    my @yyyymmdd;
    push @yyyymmdd, $yyyymmdd;
    
    while (1) {

        my ($nextNightdate) = &getNextNightdate($nightdate);
        my ($nyear, $nmonth, $nday) = split(/\-/, $nextNightdate);

        if ($month == $nmonth) {

            my ($zpDay) = &zeroPadTwoPlaces($nday);

            ($nightdate) = $nyear . '-' . $nmonth . '-' . $zpDay;
            $yyyymmdd = $year . $month . $zpDay;
    
            push @yyyymmdd, $yyyymmdd;

        } else {
            last;
        }

    }
   

    return (reverse @yyyymmdd);
}



#------------------------------------------
# Get date of next night.

sub getNextNightdate {

    my ($nightdate) = @_;

    my ($year, $month, $day) = split(/\s+|\-/, $nightdate);

    if ($day == 31) {

        if ($month == 12) {
            $day = 1;
            $month = 1;
            $year++;
        } else {
            $day = 1;
            $month++;
        }

    } elsif ($day == 30) {

        if (($month == 4) or ($month == 6) or ($month == 9) or($month == 11)) {
            $day = 1;
            $month++;
        } else {
            $day++;
        }

    } elsif ($day == 28) {

        if (($month == 2) and (! (($year == 2016) or ($year == 2020) or ($year == 2024) or ($year == 2028)))) {
            $day = 1;
            $month++;
        } else {
            $day++;
        }

    } elsif ($day == 29) {

        if (($month == 2) and (($year == 2016) or ($year == 2020) or ($year == 2024) or ($year == 2028))) {
            $day = 1;
            $month++;
        } else {
            $day++;
        }

    } else {

        $day++;

    }

    my $nextnightdate = sprintf("%4d-%02d-%02d", $year, $month, $day);

    return ($nextnightdate);
}


sub zeroPadTwoPlaces {

    my ($uniqueid) = @_;

    $uniqueid =~ s/0*(\d+)/$1/;

    if ($uniqueid < 10) {
        $uniqueid = '0' . $uniqueid;
    }

    return ($uniqueid);
}
