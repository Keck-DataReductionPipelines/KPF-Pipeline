#! /bin/csh

#
# Script to build the C software system for KPF.
#
# Russ Laher (2/14/24)

#--------Configure build environment--------------------------
setenv KPF_SW /data/user/rlaher/git/KPF-Pipeline

#--------Remove old and make new delivery directories--------------------------
cd ${KPF_SW}/c
rm -rf bin
rm -rf lib
rm -rf include
mkdir bin
mkdir lib
mkdir include
mkdir -p include/cfitsio
mkdir -p include/nan
mkdir -p include/numericalrecipes

#--------Build GSL library---------------------
echo " "     
echo "--->Building GSL 2.5 library ..."
cd ${KPF_SW}/c/common/gsl
rm -rf gsl-2.5
tar -xvf gsl-2.5.tar
cd gsl-2.5
./configure --prefix=${KPF_SW}/c
make
make check
make install
echo " "     
echo "--->Finished building GSL 2.5 library."

#--------Build cfitsio library-------------------------------
echo " "
echo "--->Building CFITSIO library, vsn 4.3.1 ..."
cd ${KPF_SW}/c/common/cfitsio/
rm -rf cfitsio-4.3.1
tar -xvf cfitsio-4.3.1.tar
mv cfitsio cfitsio-4.3.1
cd cfitsio-4.3.1
./configure --prefix=${KPF_SW}/c/common/cfitsio/cfitsio-4.3.1
make shared
make install
make imcopy
make fpack
make funpack
echo " "
echo "--->Done with CFITSIO-library make install ..."
mkdir -p ${KPF_SW}/c/include/cfitsio
cp ${KPF_SW}/c/common/cfitsio/cfitsio-4.3.1/lib/libcfits* ${KPF_SW}/c/lib
cp ${KPF_SW}/c/common/cfitsio/cfitsio-4.3.1/include/*.h ${KPF_SW}/c/include/cfitsio
cp ${KPF_SW}/c/common/cfitsio/cfitsio-4.3.1/imcopy ${KPF_SW}/c/bin
cp ${KPF_SW}/c/common/cfitsio/cfitsio-4.3.1/fpack ${KPF_SW}/c/bin
cp ${KPF_SW}/c/common/cfitsio/cfitsio-4.3.1/funpack ${KPF_SW}/c/bin
echo " "
echo "--->Finished building CFITSIO library."

#--------Build fitsverify module-------------------
echo " "
echo "--->Building fitsverify module ..."
cd ${KPF_SW}/c/common/fitsverify
rm -rf fitsverify-4.22
tar xvf fitsverify-4.22.tar
mv fitsverify fitsverify-4.22
cp Makefile fitsverify-4.22
cd fitsverify-4.22
make
echo " "
echo "--->Finished building fitsverify module."

#--------Build nan library-------------------
echo " "
echo "--->Building nan library ..."
cd ${KPF_SW}/c/common/nan
make clean
make
echo " "
echo "--->Finished building nan library."

#--------Build numericalrecipes library-------------------
echo " "
echo "--->Building numericalrecipes library ..."
cd ${KPF_SW}/c/common/numericalrecipes
make clean
make
echo " "
echo "--->Finished building numericalrecipes library."

#--------Build verifyHduSums module-------------------
echo " "
echo "--->Building verifyHduSums module ..."
cd ${KPF_SW}/c/src/verifyHduSums
make clean
make
echo " "
echo "--->Finished building verifyHduSums module."

#--------Build imheaders module-------------------
echo " "
echo "--->Building imheaders module ..."
cd ${KPF_SW}/c/src/imheaders
make clean
make
echo " "
echo "--->Finished building imheaders module."

#--------Build hdrupdate module-------------------
echo " "
echo "--->Building hdrupdate module ..."
cd ${KPF_SW}/c/src/hdrupdate
make clean
make
echo " "
echo "--->Finished building hdrupdate module."

#--------Build generateSmoothLampPattern module-------------------
echo " "
echo "--->Building generateSmoothLampPattern module ..."
cd ${KPF_SW}/c/src/generateSmoothLampPattern
make clean
make
echo " "
echo "--->Finished building generateSmoothLampPattern module."

#--------Build makeTestFitsFile module-------------------
echo " "
echo "--->Building makeTestFitsFile module ..."
cd ${KPF_SW}/c/src/makeTestFitsFile
make clean
make
echo " "
echo "--->Finished building makeTestFitsFile module."
