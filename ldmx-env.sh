#!/bin/bash

# SUGGESTION:
#   Put the following command in your .bashrc file to make setting up
#   the ldmx environment easier
#   alias ldmxenv='source <path-to-this-file>/ldmx-env.sh'

# This is the full path to the directory containing ldmx-sw
#   CHANGE ME if your ldmx-sw repo is not in the ldmx group area under your SLAC username
LDMXBASE="/nfs/slac/g/ldmx/users/$USER"

### Helpful Aliases and Bash Functions
# cmake command required to be done before make to build ldmx-sw
# WARNING: must be in $LDMXBASE/ldmx-sw/build directory when you run cmake
#   if you run it outside of build directory and it completes, 
#   you will need to git reset and git clean to remove
#   the build files that are mixed with the source files
# This can only be run after this script is sourced.
# Provide any additional options as normal after this command
#   e.g. ldmxcmake -DINSTALL_DOC=ON
function ldmxcmake {
    (set -x; cmake -DCMAKE_INSTALL_PREFIX=$LDMX_INSTALL_PREFIX -DXercesC_DIR=$XERCESDIR -DPYTHON_EXECUTABLE=`which python` -DPYTHON_INCLUDE_DIR=${PYTHONHOME}/include/python2.7 -DPYTHON_LIBRARY=$PYTHONHOME/lib/libpython2.7.so "$@" ../ ;)
}

# skips directories that aren't modules
#   now run 'grepmodules <pattern>' in ldmx-sw/ to search for a <pattern> in the module source files
alias grepmodules='grep --exclude-dir=build --exclude-dir=docs --exclude-dir=install --exclude-dir=.git -rHn'

# installation prefix for ldmx-sw
export LDMX_INSTALL_PREFIX="$LDMXBASE/ldmx-sw-2.0/install" #needed for cmake

# total remake command
# nuclear option
#   deletes ldmx install
#   goes to build directory, completely deletes old build, 
#   re-executes cmake and make, returns to prior directory
ldmxremake() {
        rm -rf $LDMX_INSTALL_PREFIX &&
        cd $LDMXBASE/ldmx-sw-2.0/build &&
        rm -r * &&
        ldmxcmake &&
        make install -j8 &&
        cd -
}

### The rest is believed to be the same for all users
# It is a hassle to change the gcc version because all of the other
# libraries could change versions. This means when we want to change
# the gcc version, you must go through the cvmfs directory tree and
# input the versions of these other libraries that we need. There is
# almost certainly an intelligent bash script that could do this for
# us, but I am not writing one write now.
# You also need to touch these library directories so that they
# appear when cmake looks for them.

# initialize required libraries for ldmx
source scl_source enable devtoolset-6

# Turn off emailing for batch jobs
export LSB_JOB_REPORT_MAIL=Y

##################################################################
# same location regardless of system
# local installations of geant4 and root
# personal build (using gcc821 and c++17)
PERSONAL_HOME="/nfs/slac/g/ldmx/users/eichl008"
G4DIR="$PERSONAL_HOME/geant4/10.2.3_v0.3-gcc821/install"
ROOTDIR="$PERSONAL_HOME/root/6.16.00-gcc821-cxx17/install"
#ROOTDIR="/nfs/slac/g/ldmx/software/root_v6-20-02/install_gcc8.3.1_cos7"
source $ROOTDIR/bin/thisroot.sh                                 #root 
source $G4DIR/bin/geant4.sh                                     #geant4

##################################################################
# location of cms shared libraries - changes depending on system
node_name_=$(uname -n) #get name of machine
if grep -q rhel <<< "$node_name_"; then
    #rhel machine
    # use this to specifiy which gcc should be used in compilation
    CVMFSDIR="/cvmfs/cms.cern.ch/slc6_amd64_gcc700"
    export XERCESDIR="$CVMFSDIR/external/xerces-c/3.1.3-fmblme" #needed for cmake
    GCCDIR="$CVMFSDIR/external/gcc/7.0.0-fmblme"
    BOOSTDIR="$CVMFSDIR/external/boost/1.67.0"
    PYTHONDIR="$CVMFSDIR/external/python/2.7.14"
    source /nfs/slac/g/ldmx/software/setup_gcc6.3.1_rhel6.sh
    export ROOTDIR=$SOFTWARE_HOME/root-6.18.04/install_gcc6.3.1_rhel6
    source $ROOTDIR/bin/thisroot.sh
    
    ## Initialize libraries/programs from cvmfs and /local/cms
    # all of these init scripts add their library paths to LD_LIBRARY_PATH
    source $CVMFSDIR/external/cmake/3.10.0/etc/profile.d/init.sh    #cmake
    source $XERCESDIR/etc/profile.d/init.sh                         #xerces-c
    source $PYTHONDIR/etc/profile.d/init.sh                         #python
    source $CVMFSDIR/external/bz2lib/1.0.6-fmblme/etc/profile.d/init.sh    #bz2lib
    source $CVMFSDIR/external/zlib/1.0-fmblme/etc/profile.d/init.sh        #zlib
    source $GCCDIR/etc/profile.d/init.sh                            #gcc
    source $BOOSTDIR/etc/profile.d/init.sh                          #boost
    source $CVMFSDIR/external/py2-xgboost/0.80/etc/profile.d/init.sh #xgboost

    # NEED DIFFERENT ROOT AND GEANT4 BUILDS USING THIS ENVIRONMENT
    echo "Rhel machines not supported! Please switch to a centos7 machine."
elif grep -q cent <<< "$node_name_"; then
    #centos machine
    # use this to specifiy which gcc should be used in compilation
    CVMFSDIR="/cvmfs/cms.cern.ch/slc7_amd64_gcc820"
    export XERCESDIR="$CVMFSDIR/external/xerces-c/3.1.3" #needed for cmake
    GCCDIR="$CVMFSDIR/external/gcc/8.2.0"
    BOOSTDIR="$CVMFSDIR/external/boost/1.67.0"
    PYTHONDIR="$CVMFSDIR/external/python/2.7.15"
    
    ## Initialize libraries/programs from cvmfs and /local/cms
    # all of these init scripts add their library paths to LD_LIBRARY_PATH
    source $CVMFSDIR/external/cmake/3.10.2/etc/profile.d/init.sh    #cmake
    source $XERCESDIR/etc/profile.d/init.sh                         #xerces-c
    source $PYTHONDIR/etc/profile.d/init.sh                         #python
    source $CVMFSDIR/external/bz2lib/1.0.6/etc/profile.d/init.sh    #bz2lib
    source $CVMFSDIR/external/zlib/1.0/etc/profile.d/init.sh        #zlib
    source $GCCDIR/etc/profile.d/init.sh                            #gcc
    source $BOOSTDIR/etc/profile.d/init.sh                          #boost
    source $CVMFSDIR/external/py2-xgboost/0.82/etc/profile.d/init.sh #xgboost
else
    echo "Unknown machine type '$node_name_'. Either 'rhel' or 'cent' should be in machine name."
fi

export PYTHONHOME=$PYTHON_ROOT #needed for cmake

# add libraries to cmake/make search path for linking
export LD_LIBRARY_PATH=$LDMX_INSTALL_PREFIX/lib:$LD_LIBRARY_PATH

# add ldmx python scripts to python search path
export PYTHONPATH=$LDMX_INSTALL_PREFIX/lib/python:$PYTHONPATH

# add ldmx executables to system search path
export PATH=$LDMX_INSTALL_PREFIX/bin:$PATH

