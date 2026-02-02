#!bin/bash
# // Usage: ./Data4Classify $DataSource $Mode $Km2aRootFile $RecRootFile $FiltedRootFile(output)
# //  ($DataSource: simulation or experiment, $mode: 0: gamma, 2: proton, 1: monopole) 

cd /home/lhaaso/zhangzhonghua/KM2AMCrec_V3/Filt_Event

DataSource=simulation
Mode=0

DATname=DAT000001
in_rootdir=/eos/user/z/zhangzhonghua/DATfile/gamma/1e4_1e5
out_rootdir=/home/lhaaso/zhangzhonghua/KM2AMCrec_V3/data/Dataset_Filted/Simulation/gamma/test
log10E_low=0.6380945
log10E_high=1.2833369

Km2aRootFile=$in_rootdir/$DATname.root
RecRootFile=$in_rootdir/rec_$DATname.root
FiltedRootFile=$out_rootdir/filted_$DATname.root

time ./bin/Data4Classify $DataSource $Mode $Km2aRootFile $RecRootFile $FiltedRootFile $log10E_low $log10E_high