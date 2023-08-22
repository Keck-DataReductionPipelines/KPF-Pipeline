#! /usr/local/bin/perl

use strict;
use warnings;


# my @mmdd=("0429","0428","0427","0516");                 # Ran 2023 May 16, morning.

#my @mmdd=("0426","0425","0424","0423","0422");           # Ran 2023 May 16, at ~3:30 p.m., but died ~6 p.m.  Had to restart again next morning at 5:10 a.m.

my @mmdd=("0517","0421","0420","0419","0418","0417");     # Ran 2023 May 17, at ~3:00 p.m.


my $scriptfile = "runit.sh";

if (! open(SCR, ">$scriptfile") ) {
  die "*** Error: Couldn't open $scriptfile for writing; quitting...\n";
}

my $shebang = '#! /bin/bash -l';

print SCR "$shebang\n";

foreach my $mmdd (@mmdd) {
  print "mmdd=$mmdd\n";

  my $outfile = "runDailyPipelines_2023${mmdd}.sh";

  if (! open(OUT, ">$outfile") ) {
    die "*** Error: Couldn't open $outfile for writing; quitting...\n";
  }

  my @op = `cat runDailyPipelines_20230430.sh`;

  foreach my $op (@op) {
    $op =~ s/0430/$mmdd/g;
    print OUT "$op";
  }

  if (! close(OUT) ) {
    die "*** Error: Couldn't close $outfile; quitting...\n";
  }

  `chmod +x $outfile`;

  my $cmd = '/data/user/rlaher/git/KPF-Pipeline/cronjobs/runDailyPipelines_20230430.sh >& /data/user/rlaher/git/KPF-Pipeline/runDailyPipelines_20230430.out';
  $cmd =~ s/0430/$mmdd/g;

  print SCR "$cmd\n";
}

if (! close(SCR) ) {
  die "*** Error: Couldn't close $scriptfile; quitting...\n";
}

`chmod +x $scriptfile`;

exit 0;
