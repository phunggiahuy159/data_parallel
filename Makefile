# =============================================================================
# Makefile for PDSDBSCAN-D (Parallel DBSCAN with Union-Find, MPI)
# =============================================================================
#
# Targets:
#   make          → build pdsdbscan (MPI parallel binary)
#   make seq      → build pdsdbscan_seq (single-process, no MPI)
#   make clean    → remove binaries and object files
#   make check    → quick single-node smoke test (4 processes)
#
# Requirements:
#   mpicxx  (OpenMPI or MPICH)
#   g++ >= 7  (C++17)

CXX      := mpicxx
CXX_SEQ  := g++

# Source fingerprint (SHA1 of all parallel sources, 12 hex chars). The runtime
# guard in main.cpp compares this across ranks and refuses to run a cluster of
# mismatched binaries. .gitattributes enforces LF so the hash matches everywhere.
BUILD_HASH := $(shell cat src/*.cpp src/*.hpp 2>/dev/null | sha1sum | cut -c1-12)

CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Isrc -DBUILD_HASH=\"$(BUILD_HASH)\"
LDFLAGS  :=

TARGET     := pdsdbscan
TARGET_SEQ := pdsdbscan_seq

SRCS := src/main.cpp \
        src/dbscan_seq.cpp \
        src/pdsdbscan.cpp

HDRS := src/union_find.hpp \
        src/kd_tree.hpp \
        src/dbscan_seq.hpp \
        src/pdsdbscan.hpp

OBJS := $(SRCS:.cpp=.o)

# ---- Default target: parallel binary ----------------------------------------
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Built: $(TARGET)"

src/%.o: src/%.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# main.o always recompiles so the embedded BUILD_HASH stays current even when
# only another translation unit changed.
src/main.o: src/main.cpp $(HDRS) FORCE
	$(CXX) $(CXXFLAGS) -c -o $@ $<
FORCE:

# ---- Sequential-only binary (for baseline timing) ---------------------------
SEQ_SRCS := src/dbscan_seq.cpp
SEQ_MAIN := tools/seq_main.cpp      # minimal wrapper, see below

seq: $(TARGET_SEQ)

$(TARGET_SEQ): src/dbscan_seq.cpp tools/seq_main.cpp $(HDRS)
	$(CXX_SEQ) $(CXXFLAGS) -o $@ src/dbscan_seq.cpp tools/seq_main.cpp

# ---- Smoke test -------------------------------------------------------------
check: $(TARGET)
	mpirun -np 4 ./$(TARGET) --n 2000 --eps 0.5 --min-pts 5 --verify
	@echo "Smoke test passed."

# ---- Cluster-wide rebuild (run on master, requires rsync access to workers) --
deploy:
	rsync -avz --exclude='.git' --exclude='benchmark/results/' . node2:~/para/
	rsync -avz --exclude='.git' --exclude='benchmark/results/' . node3:~/para/
	ssh node2 "cd ~/para && make -j4"
	ssh node3 "cd ~/para && make -j4"
	@echo "Built on all nodes."

# ---- Clean ------------------------------------------------------------------
# ---- Environment / version report (run on each node, compare the output) ----
doctor:
	@echo "host : $$(hostname)"
	@echo "os   : $$(lsb_release -ds 2>/dev/null || uname -sr)"
	@echo "g++  : $$(g++ --version | head -1)"
	@echo "mpi  : $$(mpirun --version | head -1)"
	@echo "build: $(BUILD_HASH)"

clean:
	rm -f $(OBJS) $(TARGET) $(TARGET_SEQ)

.PHONY: all seq check deploy clean doctor
