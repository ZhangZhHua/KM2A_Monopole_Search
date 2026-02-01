/*
g++ -std=c++17 -O2 -Wall \
    -Iinclude -Isrc \
    src/LHEvent.cc   src/LHEventDict.cc \
    src/KM2AEvent.cc src/KM2AEventDict.cc \
    CombineFilted.cc \
    -o ./bin/CombineFilted \
    `root-config --cflags --libs`
*/ 

#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <iostream>
#include <vector>
#include "LHEvent.h" 
// #include "EOSopen.h"
// ./CombineFilted $CombinedFile $(ls filted*.root | grep -v CombinedFile.root)
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "用法: " << argv[0] << " 输出.root 输入1.root 输入2.root ..." << std::endl;
        return 1;
    }

    std::string outfilename = argv[1];

    // 创建 TChain 用于收集多个文件中的 tree
    TChain chain("filted_tree"); // 注意这里名字必须和你原来文件中的树名字一致

    // 添加所有输入文件
    for (int i = 2; i < argc; ++i) {
        chain.Add(argv[i]);
    }

    // 设置事件对象指针并绑定 branch
    LHFiltedEvent* event = new LHFiltedEvent();
    chain.SetBranchAddress("FiltedEvent", &event);

    // 创建输出文件和树
    TFile* outfile = new TFile(outfilename.c_str(), "RECREATE");
    TTree* outtree = chain.CloneTree(0);  // 创建一个空的 clone 树结构

    // 遍历所有事件并复制到输出树中
    Long64_t nentries = chain.GetEntries();
    for (Long64_t i = 0; i < nentries; ++i) {
        chain.GetEntry(i);
        outtree->Fill();
    }

    // 写入并关闭
    outtree->Write();
    outfile->Close();

    std::cout << "合并完成，输出文件：" << outfilename << std::endl;

    delete event;
    return 0;
}