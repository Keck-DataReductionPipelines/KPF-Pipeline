#! /usr/bin/perl

use strict;
use warnings;

select STDERR; $| = 1; select STDOUT; $| = 1;


# Initialize fixed parameters and read command-line parameter.

my $iam = 'cleanOldFilesFromDisk.pl';
my $version = '2.0';


# Sandbox directory for intermediate files.
# E.g., /data/user/rlaher/sbx
# Use /sbx to execute this Perl script inside a docker container; e.g.,
# export KPFCRONJOB_SBX=/sbx
# This assumes the docker run command maps -v /data/user/rlaher/sbx:/sbx

my $sandbox = $ENV{KPFCRONJOB_SBX};

if (! (defined $sandbox)) {
    die "*** Env. var. KPFCRONJOB_SBX not set; quitting...\n";
}

my $cmd0 = "df -h ${sandbox}";
print "Executing [$cmd0]...\n";
my @op0 = `$cmd0`;
if (@op0) { print "Output from [$cmd0]=[@op0]\n"; }

my $olderthanndays = 1;   # Remove files older than 3 days.

my $dir1 = "${sandbox}/L0";
&removeOldSubDirs($olderthanndays, $dir1);

my $dir2 = "${sandbox}/2D";
&removeOldSubDirs($olderthanndays, $dir2);

my $dir3 = "${sandbox}/masters/wlpixelfiles";
&removeOldFiles($olderthanndays, $dir3);

my $dir4 = "${sandbox}/analysis";
&removeOldSubDirs($olderthanndays, $dir4);

my $dir5 = "${sandbox}/logs";
&removeOldSubDirs($olderthanndays, $dir5);

my $dir6 = "${sandbox}/masters";
&removeOnlyOldYYYYMMDDSubDirs($olderthanndays, $dir6);

my $dir7 = "${sandbox}/masters/pool";
&removeOldFiles($olderthanndays, $dir7);

my $cmd1 = "df -h ${sandbox}";

print "Executing [$cmd1]...\n";
my @op1 = `$cmd1`;
if (@op1) { print "Output from [$cmd1]=[@op1]\n"; }


#------------------------------------------
# Remove subdirectories under input directory that
# are older than N days, and all files therein.

sub removeOldSubDirs {

    my ($ndaysold, $dir) = @_;

    print "dir = $dir\n";
    opendir(DIR, "$dir");
    my @files = readdir DIR;
    closedir DIR;
    foreach my $file (@files) {
        print "file = $file\n";
        if (($file eq ".") or ($file eq "..") or ($file =~ /^\.+/)) {
          print "Skipping file---->[$file]\n";
          next;
        }
        $file = $dir . "/" . $file;
        if ((-d $file) and (! ($file =~ /^.+\/\..*$/))) {
            my $fileage = -M $file;
            if ($fileage > $ndaysold) {
                print "Removing directory---->[$file]\n";
                `rm -rf $file`;
            }
        }
    }
}


#------------------------------------------
# Remove files under input directory that
# are older than N days.

sub removeOldFiles {

    my ($ndaysold, $dir) = @_;

    print "dir = $dir\n";
    opendir(DIR, "$dir");
    my @files = readdir DIR;
    closedir DIR;
    foreach my $file (@files) {
        print "file = $file\n";
        if (($file eq ".") or ($file eq "..") or ($file =~ /^\.+/)) {
          print "Skipping file---->[$file]\n";
          next;
        }
        $file = $dir . "/" . $file;
        if ((-f $file) and (! ($file =~ /^.+\/\..*$/))) {
            my $fileage = -M $file;
            if ($fileage > $ndaysold) {
                print "Removing file---->[$file]\n";
                `rm -f $file`;
            }
        }
    }
}


#------------------------------------------
# Remove only YYYYMMDD subdirectories under input directory that
# are older than N days, and all files therein.

sub removeOnlyOldYYYYMMDDSubDirs {

    my ($ndaysold, $dir) = @_;

    print "dir = $dir\n";
    opendir(DIR, "$dir");
    my @files = readdir DIR;
    closedir DIR;
    foreach my $file (@files) {
        print "file = $file\n";
        if (($file eq ".") or ($file eq "..") or ($file =~ /^\.+/)) {
          print "Skipping file---->[$file]\n";
          next;
        }
        $file = $dir . "/" . $file;
        if ((-d $file) and ($file =~ /^.+\/\d\d\d\d\d\d\d\d$/)) {
            my $fileage = -M $file;
            if ($fileage > $ndaysold) {
                print "Removing directory---->[$file]\n";
                `rm -rf $file`;
            }
        }
    }
}




