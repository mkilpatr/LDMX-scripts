from LDMX.Framework import ldmxcfg
# Create a process with pass name "vetoana"
p=ldmxcfg.Process('ecalMultiElecAna')
from LDMX.Analysis import ecal
ecal_veto_ana = ecal.ECalMultiElecAnalyzer()
#you could change the settings if you want/need to
#ecal_veto_ana.ecal_veto_collection = 'SomeOtherEcalVetoCollection'
p.sequence = [ ecal_veto_ana ]
p.inputFiles = ["$inputEventFile" ]
p.histogramFile = "$histogramFile"
#p.pause()
