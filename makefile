CC = g++


CCFLAGS = -std=c++14
#CCFLAGS = -std=c++0x

INCLUDE_DIR = src

## Libraries: External tools, csv-parser, Eigen etc
#EIGEN_DIR = lib/eigen-3.3.8
#EIGEN_DIR = lib/eigen-3.3.9
EIGEN_DIR = ../peri-ice/lib/eigen-3.3.9
#INIH_DIR = lib/inih
#EIGEN_DIR = lib/eigen-3.2.10
#CSV_DIR = lib/fast-cpp-csv-parser

BUILD_DIR = bin
OBJ_DIR = obj
RUN_DIR = run
DATA_DIR = data
OUTPUT_DIR = output

CPPFLAGS += -O3
CPPFLAGS += -Wall -Wextra # -Wshadow # -Wcast-align # -Wold-style-cast 

CPPFLAGS += -I$(INCLUDE_DIR)
CPPFLAGS += -I$(EIGEN_DIR)
#CPPFLAGS += -I$(INIH_DIR)
#CPPFLAGS += -I /usr/include/hdf5/serial/ -lhdf5_serial -lhdf5_cpp
CPPFLAGS += -I/usr/include/hdf5/serial/ -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_serial -lhdf5_cpp
#CPPFLAGS += -I$(CSV_DIR)

#CPPFLAGS += -lpthread


# All subdirectories of includes
SUB_DIRS = $(wildcard $(INCLUDE_DIR)/*)
# Substitute 
SUB_DIRS_BUILD = $(patsubst $(INCLUDE_DIR)/%, $(OBJ_DIR)/%, $(SUB_DIRS))
# Data dirs
# All header files
T_h = $(wildcard  $(INCLUDE_DIR)/**/*.h)
# All cpp files
T_cpp = $(wildcard  $(INCLUDE_DIR)/**/*.cpp)
# .cpp replaced by .o
T_o = $(T_cpp:%.cpp=%.o)
	# Replace the include path structure by object path structure: pattern-substitute
T_o_Build = $(patsubst $(INCLUDE_DIR)/%, $(OBJ_DIR)/%, $(T_o))

#all: dir simulate2d
all: dir both

#simulate2d: $(OBJ_DIR)/read/varload.o $(OBJ_DIR)/read/load_mesh.o $(OBJ_DIR)/particle/particle.o
both: $(T_o_Build)
	@echo "Building final target in both 2d and 3d:"
	make simulate2d
	make simulate3d

worm:
	@echo "Building worm in 2d"
	$(CC) $(CCFLAGS) $(RUN_DIR)/worm.cpp $? -o $(BUILD_DIR)/$@ $(CPPFLAGS)  -fopenmp

simulate2d:
	@echo "Building in 2D"
	$(CC) $(CCFLAGS) $(RUN_DIR)/simulate2d.cpp $? -o $(BUILD_DIR)/$@ $(CPPFLAGS)  -fopenmp

simulate3d:
	@echo "Building in 3D"
	$(CC) $(CCFLAGS) $(RUN_DIR)/simulate3d.cpp $? -o $(BUILD_DIR)/$@ $(CPPFLAGS)  -fopenmp

debug: $(T_o_Build)
	@echo "Building final target in 2D for debug:"
	#$(CC) $(CCFLAGS) $(RUN_DIR)/simulate2d.cpp $? -o $(BUILD_DIR)/$@ $(CPPFLAGS)  -fopenmp
	$(CC) $(CCFLAGS) $(RUN_DIR)/simulate2d.cpp $? -o $(BUILD_DIR)/$@ $(CPPFLAGS) -g  -fopenmp
	gdb $(BUILD_DIR)/simulate2d

test: $(T_o_Build)
	@echo "Building test target:"
	$(CC) $(CCFLAGS) $(RUN_DIR)/test.cpp $? -o $(BUILD_DIR)/$@ $(CPPFLAGS)
	@echo "Running test executable:"
	@echo "-----------------------------------------------------------------------"
	@$(BUILD_DIR)/test

# Maintain the same directory structure
$(OBJ_DIR)/%.o: $(INCLUDE_DIR)/%.cpp

	@echo "Building:"
	$(CC) $(CCFLAGS) -c $? -o $@ $(CPPFLAGS)

ex2:
	@echo "Running executable 2D:"
	@echo "-----------------------------------------------------------------------"
	@time -p $(BUILD_DIR)/simulate2d

ex3:
	@echo "Running executable 3D:"
	@echo "-----------------------------------------------------------------------"
	@time -p $(BUILD_DIR)/simulate3d

exw:
	@echo "Running executable worm:"
	@echo "-----------------------------------------------------------------------"
	@time -p $(BUILD_DIR)/worm

clean:
	@echo "Deleting:"
	rm -rf $(BUILD_DIR)/* $(OBJ_DIR)/*

genplot:
	@echo "Generating plots:"
	@#python3 write_plot.py
	#python3 hdf_plot.py
	#python3 hdf_plot_cl.py
	python3 plot_current.py
	sxiv $(OUTPUT_DIR)/img/* &

genvid:
	@#RNDN=$(date +%F-%H-%M)$(cat /dev/urandom | tr -cd 'a-z0-9' | head -c 5);

	RNDN=$$(date +%F-%H-%M-%S)-$$(cat /dev/urandom | tr -cd 'a-z0-9' | head -c 5); \
       	ffmpeg -i $(OUTPUT_DIR)/img/img_tc_%5d.png -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -r 25 -crf 30 $(OUTPUT_DIR)/vid/$${RNDN}.mp4; \
	#mplayer -loop 0 $(OUTPUT_DIR)/vid/$${RNDN}.mp4; \
	echo "Saved video to file:" $(OUTPUT_DIR)/vid/$$RNDN.mp4;

showvid:
	@echo "Playing video:"
	mplayer -loop 0 $(OUTPUT_DIR)/vid/$$(ls -t $(OUTPUT_DIR)/vid | head -1)
	@echo "Video file: " $(OUTPUT_DIR)/vid/$$(ls -t $(OUTPUT_DIR)/vid | head -1)


wall_reaction:
	@echo "Generating wall reaction file:"
	python3 gen_wall_reaction.py
	@echo "Plottin wall reaction:"
	python3 plot_wall_reaction.py
	sxiv $(OUTPUT_DIR)/img/* &

showplot:
	sxiv $(OUTPUT_DIR)/img/*.png &

clout:
	@echo "Deleting output files"
	rm -rf $(OUTPUT_DIR)/csv/*.csv $(OUTPUT_DIR)/hdf5/*.h5 $(OUTPUT_DIR)/img/*.png
	rm -rf $(OUTPUT_DIR)/img/*.mp4
	rm -r $(OUTPUT_DIR)/*.npy

getfresh_py:
	@echo "Delete exiting data"
	rm -rf $(DATA_DIR)/*
	@echo "Getting new data"
	mkdir -p $(DATA_DIR)/csv $(DATA_DIR)/hdf5
	##cp ~/mperidem/data/csv/* $(DATA_DIR)/
	#cp ~/mperidem/data/hdf5/* $(DATA_DIR)/hdf5/
	cp meshdata/all.h5 $(DATA_DIR)/hdf5/

setup:
	@echo "Generating new setup"
	python3 gen_setup.py
	make getfresh_py

getfresh_mat:
	@echo "Delete exiting data"
	rm -rf $(DATA_DIR)/*
	@echo "Getting new data"
	mkdir -p $(DATA_DIR)/csv $(DATA_DIR)/hdf5
	##cp ~/mperidem/data/csv/* $(DATA_DIR)/
	cp ~/testzone/mperidem/data/hdf5/* $(DATA_DIR)/hdf5/
	#cp meshdata/all.h5 $(DATA_DIR)/hdf5/

testmpi:
	@echo "Building test mpi target:"
	mpic++ $(CCFLAGS) $(RUN_DIR)/testmpi.cpp $? -o $(BUILD_DIR)/$@ $(CPPFLAGS)

	@echo "Running test mpi executable with mpirun:"
	@echo "-----------------------------------------------------------------------"
	@mpirun $(BUILD_DIR)/testmpi

testomp:
	@echo "Building test omp target:"
	$(CC) $(CCFLAGS) $(RUN_DIR)/testomp.cpp $? -o $(BUILD_DIR)/$@ $(CPPFLAGS) -fopenmp

	@echo "Running test omp executable:"
	@echo "-----------------------------------------------------------------------"
	@time -p $(BUILD_DIR)/testomp


dir:
	@echo "Creating directories:"
	mkdir -p $(BUILD_DIR) $(OBJ_DIR) $(SUB_DIRS_BUILD)
	mkdir -p $(DATA_DIR)/csv $(DATA_DIR)/hdf5
	mkdir -p $(OUTPUT_DIR)/csv $(OUTPUT_DIR)/hdf5 $(OUTPUT_DIR)/img $(OUTPUT_DIR)/vid

again: clean $(TARGETS)  

