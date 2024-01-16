

CXX		= mpicxx -std=c++17
CFLAGS	= -O3 -m64 -Wall -fopenmp -I. -I"${MKLROOT}/include"
LDFLAGS	=  -L${MKLROOT}/lib -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

ODIR	= ./obj
LDIR	= ./lib

DEPS	= basis.hpp build_tree.hpp comm.hpp geometry.hpp kernel.hpp
objs	= main.o basis.o build_tree.o comm.o kernel.o
OBJ = $(patsubst %,$(ODIR)/%,$(objs))

$(ODIR)/%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CFLAGS)

main: $(OBJ)
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ main $(INCDIR)/*~
