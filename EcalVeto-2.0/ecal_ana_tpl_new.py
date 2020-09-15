
from LDMX.Framework import ldmxcfg

# Create a process with pass name "vetoana"
p=ldmxcfg.Process('ecalMultiElecAna')

# Append the library that contains the analyzer below to the list of libraries 
# that the framework will load.
# change the next line to point to your own ldmx-sw install directory!
#p.libraries.append("/nfs/slac/g/ldmx/users/vdutta/ldmx-analysis-2.1/install/lib/libAnalysis.so")
# Should now be done automatically when declaring module below

# Create an instance of the ECal veto analyzer.  This analyzer is used to create
# an ntuple out of ECal BDT variables. The analyzer requires that the
# veto collection name be set.   
ecal_veto_ana = ldmxcfg.Producer("ecalMulti", "ldmx::ECalMultiElecAnalyzer", "Analysis")
ecal_veto_ana.ecal_simhit_collection = "EcalSimHits"

from LDMX.DetDescr import EcalHexReadout
ecal_veto_ana.hexReadout = EcalHexReadout.EcalHexReadout()

# Define the order in which the analyzers will be executed.
p.sequence=[ecal_veto_ana]

p.keep = [ "keep .*" ]

# Specify the list of input files. 
p.inputFiles=["$inputEventFile"]

# Specify the output file.  When creating ntuples or saving histograms, the 
# output file name is specified by setting the attribute histogramFile.  
p.histogramFile="$histogramFile"

# Print out the details of the configured analyzers. 
print(p)

