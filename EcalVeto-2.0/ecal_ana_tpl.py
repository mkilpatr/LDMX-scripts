
from LDMX.Framework import ldmxcfg

# Create a process with pass name "vetoana"
p=ldmxcfg.Process('vetoana')

# Append the library that contains the analyzer below to the list of libraries 
# that the framework will load.
p.libraries.append("/nfs/slac/g/ldmx/users/vdutta/ldmx-analysis/install/lib/libAnalysis.so")

# Create an instance of the ECal veto analyzer.  This analyzer is used to create
# an ntuple out of ECal BDT variables. The analyzer requires that the
# veto collection name be set.   
ecal_veto_ana = ldmxcfg.Producer("ecal", "ldmx::ECalVetoAnalyzer")
ecal_veto_ana.parameters['ecal_veto_collection'] = "EcalVeto"
ecal_veto_ana.parameters['hcal_veto_collection'] = "HcalVeto"
ecal_veto_ana.parameters['tracker_veto_collection'] = "TrackerVeto"
ecal_veto_ana.parameters['trig_result_collection'] = "Trigger"
ecal_veto_ana.parameters['ecal_simhit_collection'] = "EcalSimHits"
ecal_veto_ana.parameters['ecal_rechit_collection'] = "EcalRecHits"

# Define the order in which the analyzers will be executed.
p.sequence=[ecal_veto_ana]

p.keep = [ "keep .*" ]

# Specify the list of input files. 
p.inputFiles=["$inputEventFile"]

# Specify the output file.  When creating ntuples or saving histograms, the 
# output file name is specified by setting the attribute histogramFile.  
p.histogramFile="$histogramFile"

# Print out the details of the configured analyzers. 
#print p
