CXX=g++
CXXFLAGS:= -std=c++14 -Ofast -Wall -Werror -Wextra -pedantic
DEBUGS:= -std=c++14 -g3
PROFIL:= -pg
LIBS:= `pkg-config --libs opencv` -lboost_system -lboost_filesystem -ltbb
SRC:= ./src
SOURCES:= $(wildcard $(SRC)/*.cc)
OBJS:= $(addsuffix .o, $(basename $(SOURCES)))
EXEC:= fight_detection

all : $(OBJS) $(EXEC)

$(EXEC): $(OBJS)
	$(CXX)  $(OBJS) $(CXXFLAGS) $(LIBS) -o $(EXEC)

debug:
	$(CXX)  $(SOURCES) $(DEBUGS) $(LIBS) -o $(EXEC)_debug

profil:
	$(CXX)  $(SOURCES) $(CXXFLAGS) $(PROFIL) $(LIBS) -o $(EXEC)_prof

.PHONY: clean

clean:
	rm -f $(EXEC)
	rm -f $(EXEC)_debug

full_clean: clean clean_obj
	rm -r features
	rm -r SVM
	rm totalFeatures.csv
	rm vocabulary.csv

clean_obj:
	rm -f $(SRC)/*.o

test: clean_obj clean all
	./test
