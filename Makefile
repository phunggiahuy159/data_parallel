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
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Isrc
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
clean:
	rm -f $(OBJS) $(TARGET) $(TARGET_SEQ)

.PHONY: all seq check deploy clean
