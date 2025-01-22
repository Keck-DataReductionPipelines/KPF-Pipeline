#! /usr/local/bin/perl

use strict;
use warnings;

select STDERR; $| = 1; select STDOUT; $| = 1;


# Sandbox directory for intermediate files.
# E.g., /data/user/rlaher/sbx
my $sandbox = $ENV{KPFCRONJOB_SBX};

if (! (defined $sandbox)) {
    die "*** Env. var. KPFCRONJOB_SBX not set; quitting...\n";
}


my $cmd0 = "df -h ${sandbox}";
print "Executing [$cmd0]...\n";
my @op0 = `$cmd0`;
if (@op0) { print "Output from [$cmd0]=[@op0]\n"; }

my $olderthanndays = 3;   # Remove files older than 3 days.

my $dir1 = "${sandbox}/L0";
&removeOldSubDirs($olderthanndays, $dir1);


my $dir2 = "${sandbox}/2D";
&removeOldSubDirs($olderthanndays, $dir2);

my $dir3 = "${sandbox}/masters/wlpixelfiles";
&removeOldFiles($olderthanndays, $dir3);

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
                `rm -rf $file`;
            }
        }
    }
}




