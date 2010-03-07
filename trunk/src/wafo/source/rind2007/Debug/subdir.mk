################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
F_SRCS += \
../fimod.f \
../intmodule.f \
../jacobmod.f \
../rindmod2007.f \
../swapmod.f \
../test_fimod.f \
../test_rindmod2007.f 

OBJS += \
./fimod.o \
./intmodule.o \
./jacobmod.o \
./rindmod2007.o \
./swapmod.o \
./test_fimod.o \
./test_rindmod2007.o 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.f
	@echo 'Building file: $<'
	@echo 'Invoking: GNU Fortran Compiler'
	gfortran -funderscoring -O0 -g -Wall -c -fmessage-length=0 -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

fimod.o: ../fimod.f

intmodule.o: ../intmodule.f

jacobmod.o: ../jacobmod.f

rindmod2007.o: ../rindmod2007.f FIMOD.o RCRUDEMOD.o JACOBMOD.o SWAPMOD.o

swapmod.o: ../swapmod.f

test_fimod.o: ../test_fimod.f FIMOD.o

test_rindmod2007.o: ../test_rindmod2007.f FIMOD.o


