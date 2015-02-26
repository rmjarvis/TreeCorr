# Set this to whatever compiler you want to use:
CC= g++
#CC= /Users/Mike/clang/build/Release+Asserts/bin/clang++
#CC= icpc

# Set this to the correct OpenMP flag for your compiler.
# For g++, it is -fopenmp.
# For icpc, it is -openmp
# To disable OpenMP, just comment out this line.
# For clang++, you need to comment it out, since they don't support OpenMP.
#OMP= -fopenmp
#OMP= -openmp

# Set this to the base directory for your cfitsio installation.
# i.e. fitsio.h should be in $(FITSDIR)/include/ and 
#      libcfitsio.a should be in $(FITSDIR)/lib/
#FITSDIR= /usr/local
FITSDIR= /sw
#FITSDIR= /usr/local/cfitsio
#FITSDIR= /usr/global/cfitsio

# Set this to the directory where you want the executables to be placed.
BINDIR= .
#BINDIR= /Users/Mike/bin
#BINDIR= /home/mjarvis/bin

# You shouldn't normally need to edit anything below this point.
INCLUDE= -I. -I$(FITSDIR)/include
LIBS= $(OMP) -L$(FITSDIR)/lib -lcfitsio -lm
CFLAGS= $(INCLUDE) $(OMP) -O3 -fno-strict-aliasing -Wall -Werror -g3
ODIR= .odir

#all : $(BINDIR)/corr2 $(BINDIR)/make_tests $(BINDIR)/corrnnna $(BINDIR)/correeea $(BINDIR)/correeeb $(BINDIR)/e3_m3
all : $(BINDIR)/corrnnna $(BINDIR)/correeea $(BINDIR)/correeeb $(BINDIR)/e3_m3

clean : 
	rm $(ODIR)/*.o

$(BINDIR)/corr2: $(ODIR)/Corr2.o $(ODIR)/CorrIO.o $(ODIR)/CalcT.o $(ODIR)/ConfigFile.o $(ODIR)/InputFile.o $(ODIR)/BinData2.o $(ODIR)/Cell.o $(ODIR)/Field.o $(ODIR)/BinnedCorr2.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

$(BINDIR)/make_tests: make_tests.cpp
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

$(ODIR)/Corr2.o: Corr2.cpp Cell.h Field.h CorrIO.h Split.h BinnedCorr2.h ConfigFile.h BinData2.h InputFile.h Bounds.h Split.h valid_keys.h
	$(CC) $(CFLAGS) -c Corr2.cpp -o $@

$(ODIR)/CorrIO.o: CorrIO.cpp CorrIO.h BinData2.h Cell.h InputFile.h Bounds.h
	$(CC) $(CFLAGS) -c CorrIO.cpp -o $@

$(ODIR)/CalcT.o: CalcT.cpp CalcT.h
	$(CC) $(CFLAGS) -c CalcT.cpp -o $@

$(ODIR)/ConfigFile.o: ConfigFile.cpp ConfigFile.h
	$(CC) $(CFLAGS) -c ConfigFile.cpp -o $@

$(ODIR)/InputFile.o: InputFile.cpp ConfigFile.h InputFile.h Angle.h
	$(CC) $(CFLAGS) -c InputFile.cpp -o $@

$(ODIR)/BinData2.o: BinData2.cpp BinData2.h Cell.h InputFile.h Bounds.h
	$(CC) $(CFLAGS) -c BinData2.cpp -o $@

$(ODIR)/Cell.o: Cell.cpp Cell.h  InputFile.h Bounds.h
	$(CC) $(CFLAGS) -c Cell.cpp -o $@

$(ODIR)/Field.o: Field.cpp Field.h Cell.h InputFile.h Bounds.h
	$(CC) $(CFLAGS) -c Field.cpp -o $@

$(ODIR)/BinnedCorr2.o: BinnedCorr2.cpp BinnedCorr2.h Split.h Field.h Cell.h
	$(CC) $(CFLAGS) -c BinnedCorr2.cpp -o $@

# The old programs for nnn, eee, etc.

$(BINDIR)/e3_m3: $(ODIR)/E3_M3.o $(ODIR)/OldCorrIO.o $(ODIR)/CalcT.o
	$(CC) $(CFLAGS) $(ODIR)/E3_M3.o $(ODIR)/OldCorrIO.o $(ODIR)/CalcT.o -lm -o $(BINDIR)/e3_m3

$(BINDIR)/corrnnna: $(ODIR)/CorrNNNa.o $(ODIR)/OldCorrIO.o $(ODIR)/CalcT.o
	$(CC) $(CFLAGS) $(ODIR)/CorrNNNa.o $(ODIR)/OldCorrIO.o $(ODIR)/CalcT.o -lm -o $(BINDIR)/corrnnna

$(BINDIR)/correeea: $(ODIR)/CorrEEEa.o $(ODIR)/OldCorrIO.o $(ODIR)/CalcT.o
	$(CC) $(CFLAGS) $(ODIR)/CorrEEEa.o $(ODIR)/OldCorrIO.o $(ODIR)/CalcT.o -lm -o $(BINDIR)/correeea

$(BINDIR)/correeeb: $(ODIR)/CorrEEEb.o $(ODIR)/OldCorrIO.o $(ODIR)/CalcT.o
	$(CC) $(CFLAGS) $(ODIR)/CorrEEEb.o $(ODIR)/OldCorrIO.o $(ODIR)/CalcT.o -lm -o $(BINDIR)/correeeb

$(BINDIR)/corrneea: $(ODIR)/CorrNEEa.o $(ODIR)/OldCorrIO.o $(ODIR)/CalcT.o
	$(CC) $(CFLAGS) $(ODIR)/CorrNEEa.o $(ODIR)/OldCorrIO.o $(ODIR)/CalcT.o -lm -o $(BINDIR)/corrneea

$(ODIR)/E3_M3.o: E3_M3.cpp OldCorrIO.h OldCorr.h Form.h
	$(CC) $(CFLAGS) -c E3_M3.cpp -o $(ODIR)/E3_M3.o

$(ODIR)/CorrNNNa.o: CorrNNNa.cpp OldCell.h OldCorrIO.h OldCorr.h Process3.h
	$(CC) $(CFLAGS) -c CorrNNNa.cpp -o $(ODIR)/CorrNNNa.o

$(ODIR)/CorrEEEa.o: CorrEEEa.cpp OldCell.h OldCorrIO.h OldCorr.h Process3.h
	$(CC) $(CFLAGS) -c CorrEEEa.cpp -o $(ODIR)/CorrEEEa.o

$(ODIR)/CorrEEEb.o: CorrEEEb.cpp OldCell.h OldCorrIO.h OldCorr.h PairCell.h Process2.h
	$(CC) $(CFLAGS) -c CorrEEEb.cpp -o $(ODIR)/CorrEEEb.o

$(ODIR)/CorrEE_multi.o: CorrEE_multi.cpp OldCell.h OldCorrIO.h OldCorr.h Process2.h
	$(CC) $(CFLAGS) -c CorrEE_multi.cpp -o $(ODIR)/CorrEE_multi.o

$(ODIR)/OldCorrIO.o: OldCorrIO.cpp OldCell.h OldCorrIO.h OldCorr.h
	$(CC) $(CFLAGS) -c OldCorrIO.cpp -o $(ODIR)/OldCorrIO.o

