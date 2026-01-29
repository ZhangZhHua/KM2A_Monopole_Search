# 编译器与 ROOT 设置
CXX := g++
ROOTCFLAGS := $(shell root-config --cflags)
ROOTLIBS := $(shell root-config --libs)
CXXFLAGS := -O2 -Wall -fPIC -std=c++17 $(ROOTCFLAGS)
LDFLAGS := $(ROOTLIBS)

# 路径设置
ROOTDIR := $(shell pwd)
INCDIR := $(ROOTDIR)/include
SRCDIR := $(ROOTDIR)/src
OBJDIR := $(ROOTDIR)/obj
DICTDIR := $(SRCDIR)
BINDIR := $(ROOTDIR)/bin

# 源文件
SRCS := $(SRCDIR)/LHEvent.cc Data4Classify.cc
OBJS := $(OBJDIR)/Data4Classify.o $(OBJDIR)/LHEvent.o \
		$(OBJDIR)/KM2AEvent.o $(OBJDIR)/KM2AEventDict.o \
        $(OBJDIR)/LHEventDict.o

TARGET := $(BINDIR)/Data4Classify
PCMFILE := $(BINDIR)/LHEventDict_rdict.pcm $(BINDIR)/KM2AEventDict_rdict.pcm 
LIBTARGET := $(BINDIR)/libLHEvent.so

# 默认目标
all:   $(BINDIR) $(OBJDIR) $(TARGET) $(PCMFILE) $(LIBTARGET)

# 创建目录
$(BINDIR) $(OBJDIR):
	mkdir -p $@

# 生成可执行文件
$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

# 生成 ROOT 字典 - 关键修复：使用绝对路径
# ========== LHEvent 字典 ==========
$(DICTDIR)/LHEventDict.cc: $(INCDIR)/LHEvent.h $(INCDIR)/LHEventLinkDef.h
	rootcling -f $@ -rmf $(BINDIR)/LHEventDict_rdict.pcm \
   		-I$(INCDIR) $(INCDIR)/LHEvent.h $(INCDIR)/LHEventLinkDef.h
	cp -f $(SRCDIR)/LHEventDict_rdict.pcm $(BINDIR)/LHEventDict_rdict.pcm

# ========== KM2AEvent 字典 ==========
$(DICTDIR)/KM2AEventDict.cc: $(INCDIR)/KM2AEvent.h $(INCDIR)/KM2AEventLinkDef.h
	rootcling -f $@ -rmf $(BINDIR)/KM2AEventDict_rdict.pcm \
		-I$(INCDIR) $(INCDIR)/KM2AEvent.h $(INCDIR)/KM2AEventLinkDef.h
	cp -f $(SRCDIR)/KM2AEventDict_rdict.pcm $(BINDIR)/KM2AEventDict_rdict.pcm

# 编译源文件
$(OBJDIR)/LHEvent.o: $(SRCDIR)/LHEvent.cc
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $< -o $@

$(OBJDIR)/LHEventDict.o: $(DICTDIR)/LHEventDict.cc
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $< -o $@

$(OBJDIR)/KM2AEvent.o: $(SRCDIR)/KM2AEvent.cc
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $< -o $@

$(OBJDIR)/KM2AEventDict.o: $(DICTDIR)/KM2AEventDict.cc
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $< -o $@

$(OBJDIR)/Data4Classify.o: Data4Classify.cc
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $< -o $@


# 动态库链接（如用于 Python/ROOT 加载）
$(LIBTARGET): $(OBJDIR)/LHEvent.o $(OBJDIR)/LHEventDict.o \
              $(OBJDIR)/KM2AEvent.o $(OBJDIR)/KM2AEventDict.o 
	$(CXX) -shared -fPIC -o $@ $^ $(LDFLAGS)

# 清理
clean:
	rm -rf $(OBJDIR) $(TARGET) \
	$(DICTDIR)/LHEventDict.* $(DICTDIR)/KM2AEventDict.*  \
	$(DICTDIR)/LHEventDict_rdict.pcm $(BINDIR)/LHEventDict_rdict.pcm  \
	$(DICTDIR)/KM2AEventDict_rdict.pcm  $(BINDIR)/KM2AEventDict_rdict.pcm  \
	$(BINDIR)/libLHEvent.so

.PHONY: all clean