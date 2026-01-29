#!bin/bash
# ./CombineFilted $CombinedFile $(ls filted*.root | grep -v CombinedFile.root)

cd /Users/macbook/PycharmProjects/pythonProject/searchMonopole/paper/LHAASO_Monopole/Filt_Event

FiltedFileDir=/Users/macbook/PycharmProjects/pythonProject/searchMonopole/paper/LHAASO_Monopole/Filt_Event/Dataset_Filted/Simulation/gamma/test
FiltedFiles=$FiltedFileDir/filted*.root

outputfile=combined.root
CombinedFile=$FiltedFileDir/$outputfile

time ./bin/CombineFilted $CombinedFile $(ls $FiltedFiles | grep -v CombinedFile.root)

# ls /home/lhaaso/zhangzhonghua/KM2AMCrec_V3/data/Dataset_Filted/Simulation/gamma/test/filted*.root | grep -v CombinedFile.root


#!bin/bash
# ./CombineFilted $CombinedFile $(ls filted*.root | grep -v CombinedFile.root)


DIR=$(pwd)
cd $DIR
Particle="gamma"  # 或者 "monopole"
if [ "$Particle" = "gamma" ]; then
    Edir="1e3_1e4"  # 可以是 1e3_1e4, 1e4_1e5, 1e5_1e6, 1e6_1e7
    out_rootdir="$DIR/Dataset_Filted/Simulation/$Particle/$Edir"
elif [ "$Particle" = "monopole" ]; then
    Edir="E1e9"
    out_rootdir="$DIR/Dataset_Filted/Simulation/$Particle/$Edir"
else
    echo "错误: 不支持的粒子类型: $Particle"
    exit 1
fi

outputfile=combined_total.root
CombinedFile=$out_rootdir/$outputfile
echo "合并:"
echo "$(ls "$out_rootdir" | grep -v $outputfile)"

time ./bin/CombineFilted $CombinedFile $(ls "$out_rootdir"/*.root | grep -v $outputfile)

# ls /home/lhaaso/zhangzhonghua/KM2AMCrec_V3/data/Dataset_Filted/Simulation/gamma/test/filted*.root | grep -v CombinedFile.root