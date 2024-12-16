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

# Code directory of KPF-Pipeline git repo where the docker run command is executed.
# E.g., /data/user/rlaher/git/KPF-Pipeline
my $codedir = $ENV{KPFCRONJOB_CODE};

if (! (defined $codedir)) {
    die "*** Env. var. KPFCRONJOB_CODE not set; quitting...\n";
}


&updateSoftwareWithGitPull($codedir);

my $privaterefdir = $sandbox . '/reference_fits';
my $publicrefdir = '/data/kpf/reference_fits';
&updateReferenceFits($privaterefdir,$publicrefdir);

my $privaterefdir2 = $sandbox . '/reference';
my $publicrefdir2 = '/data/kpf/reference';
&updateReference($privaterefdir2,$publicrefdir2);

exit 0;


#------------------------------------------
# Update software with git pull.

sub updateSoftwareWithGitPull {

    my ($workdir) = @_;


    # Change directory.

    print "Changing to workdir = $workdir\n";

    if (! chdir "$workdir") {
        print "*** Warning: Could not change to $workdir; sleep for 10 seconds and try again...\n";
        sleep(10);
        if (! chdir "$workdir") {
            die "*** Error: Could not change to $workdir; quitting...\n";
        }
    }

    my $cmd0 = "git pull";
    print "Executing [$cmd0]...\n";
    my @op0 = `$cmd0`;
    if (@op0) { print "Output from [$cmd0]=[@op0]\n"; }
}


#------------------------------------------
# Update files in private reference_fits.

sub updateReferenceFits {

    my ($dir,$dir2) = @_;


    # Change to private directory.

    print "Changing to private reference_fits directory = $dir\n";

    if (! chdir "$dir") {
        print "*** Warning: Could not change to $dir; sleep for 10 seconds and try again...\n";
        sleep(10);
        if (! chdir "$dir") {
            die "*** Error: Could not change to $dir; quitting...\n";
        }
    }

    print "dir = $dir\n";
    opendir(DIR, "$dir");
    my @files = readdir DIR;
    closedir DIR;

    print "dir2 = $dir2\n";
    opendir(DIR, "$dir2");
    my @files2 = readdir DIR;
    closedir DIR;

    foreach my $file2 (@files2) {
        print "file2 = $file2\n";
        if (($file2 eq ".") or ($file2 eq "..") or ($file2 =~ /^\.+/)) {
          print "1 Skipping file2---->[$file2]\n";
	  next;
        }

	# Check if file2 exists in private (current) directory.
	# If not, copy it from the public directory.

        if (-e $file2) {
            print "$file2 already exists in private directory; skipping...\n";
	} else {
	    $file2 = $dir2 . "/" . $file2;

            my $cmd0 = "cp -p $file2 .";
            print "Executing [$cmd0]...\n";
            my @op0 = `$cmd0`;
            if (@op0) { print "Output from [$cmd0]=[@op0]\n"; }
        }
    }
}


#------------------------------------------
# Update files in private reference.

sub updateReference {

    my ($dir,$dir_public) = @_;


    # Change to private directory.

    print "Changing to private reference directory = $dir\n";

    if (! chdir "$dir") {
        print "*** Warning: Could not change to $dir; sleep for 10 seconds and try again...\n";
        sleep(10);
        if (! chdir "$dir") {
            die "*** Error: Could not change to $dir; quitting...\n";
        }
    }

    my $junk_obs = $dir_public . "/" . 'Junk_Observations_for_KPF.csv';

    # Copy command will leave last-updated timestamp.
    my $cmd0 = "cp $junk_obs .";
    print "Executing [$cmd0]...\n";
    my @op0 = `$cmd0`;
    if (@op0) { print "Output from [$cmd0]=[@op0]\n"; }
}
