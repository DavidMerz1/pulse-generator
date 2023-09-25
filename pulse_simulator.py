import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import subprocess
import hdf5plugin

#--------------------------------------------------------------------------------------------------

#Defines the pulse generating function

#use the upper return-commands for linear baseline and the lower return-commands for an exponential baseline (in the if-clause and the second elif-clause)


def pulse_generator(x, amplitude, decay, time_rise, offset, baseline_length): 
    if baseline_length <= x <= time_rise+baseline_length:
        return amplitude *(1/(time_rise)) * (x-baseline_length) +offset
        #return (amplitude-(baseline_start* math.exp(-baseline_decay * baseline_length)))*(1/(time_rise)) * (x-baseline_length) +offset+(baseline_start* math.exp(-baseline_decay * baseline_length))
    elif x > time_rise+baseline_length:
        return amplitude * math.exp(-decay * (x - time_rise-baseline_length)) +offset
    elif 0 <= x < baseline_length:
        return (offset+baseline_start)-(x/baseline_length)*baseline_start
        #return baseline_start* math.exp(-baseline_decay * x) +offset
    else:
        return 0

#--------------------- Enter parameters here!!! -------------------------------------------------

amplitude = 5 #amplitude of signal over offset (total amplitude is amplitude+offset!)
time_rise=0.000012 #time interval, in which graph raises from offset to the max. amplitude
decay = 10000 #determines how fast the signal will fall off after reaching max.amplitude
offset=10 #base Level, when there is no pulse

baseline_length=0.00001 #length of baseline before pulse is starting
baseline_start=0 #start value of baseline over the offset (total start value is amplitude+baseline_start)
baseline_decay=100000 #decay of baseline (only used for exponential baseline)

frequency=50 #frequency of generated pulse in Hz
simulation_time=2 #total running time of the simulation in s

#------------------------------------------------------------------------------------------------

#Definitions for calling and storing the data

filename = f"pulsedata_{amplitude}_{time_rise*1000000}_{decay}_{offset}_{frequency}_{simulation_time}_{baseline_length*1000000}.h5" #name of the hdf5-file which has the pulse data
input_folder ="/home/dbm50/gemini/input_files/" #folder where the pulse file will be stored
input_path = input_folder+filename
input_plot_filename = f"pulseplot_{amplitude}_{time_rise*1000000}_{decay}_{offset}_{frequency}_{simulation_time}_{baseline_length*1000000}.jpg" #name of the jpg-file of the pulse plot stored
input_plot_folder ="/home/dbm50/gemini/input_files/plots/"  
input_plot_path = input_plot_folder+input_plot_filename

output_filename = f"outputdata_{amplitude}_{time_rise*1000000}_{decay}_{offset}_{frequency}_{simulation_time}_{baseline_length*1000000}.h5" #name of the hdf5-file which is created by the processing routine
output_folder = "/home/dbm50/gemini/output_files/" #folder where the file created through the processing routine will be stored
output_path = output_folder+output_filename
output_plot_filename = f"outputplot_{amplitude}_{time_rise*1000000}_{decay}_{offset}_{frequency}_{simulation_time}_{baseline_length*1000000}.jpg"
output_plot_folder = "/home/dbm50/gemini/output_files/plots/"
output_plot_path = output_plot_folder+output_plot_filename


#command for running the JPROC-processing routine
command = f"/home/dbm50/julia-1.9.2/bin/julia apply_config.jl --config /home/dbm50/gemini/configfiles/dsp_channelConfig.json --input {input_path} --out_file {output_path}" 

#------------------------------------------------------------------------------------------------

#Definitions for start and end of one pulse, in total 22000 data-tupels are created per pulse!

min_x = 0.0000000000000000000001 
max_x = 0.000176 
steplength = 0.000000008 


float_pulse_number=simulation_time*frequency 
total_pulse_number=math.ceil(float_pulse_number) #rounds up the float_pulse_number to an integer

#------------------------------------------------------------------------------------------------

# Iteration variables for the while loops

frequency_counter=0 #Increases per loop by the period determined by the frequency
pulse_counter=0 #counter for filling the several collumns of the data matrix 


#-------------------------------------------------------------------------------------------------

# Creates the arrays for storing the data

values_array = np.empty((total_pulse_number+1, 22001), dtype=float) 
#Matrix with the signal values of one pulse in every column

timestamps_array = np.empty(total_pulse_number+1, dtype=float) 
#Array with the timestamp values for every pulse (timestamp = time of first signal value over offset)

#--------------------------------------------------------------------------------------------------

#Data is created here!

if frequency >= 1/max_x: #program can't simulate overlapping signals and would still create single pulses
    print('WARNING: Frequency is too high, pulses will overlap!')
    exit()

while frequency_counter <= simulation_time: #Loop generates one pulse with one cycle

    sweep_counter = 0 
    sweep_x = min_x 
    timestamp = min_x+baseline_length+frequency_counter
    timestamps_array[pulse_counter] = timestamp

    while sweep_x <= max_x: #Loop creates the data of one pulse (one value per run of the loop)
        value = pulse_generator(sweep_x, amplitude, decay, time_rise, offset, baseline_length) #value is calculated
        values_array[pulse_counter, sweep_counter] = value #value is stored in numpy-array, all the values of one pulse are stored in one row
        sweep_x += steplength 
        sweep_counter += 1 
        
            
    frequency_counter+=1/frequency
    pulse_counter+=1 
    print("Pulse ",pulse_counter," of ",total_pulse_number,"simulated.")


#------------------------------------------------------------------------------------------------

#Plots one pulse of the created pulse data!

print(timestamps_array)

column_index = 4 #number of pulse which will get plotted
data_column = values_array[column_index, :]

plt.plot(data_column)
plt.xlabel("time step")
plt.ylabel("signal")
plt.title(input_plot_filename)
plt.savefig(input_plot_path)
plt.show()

#-----------------------------------------------------------------------------------------------

#Saves the data in an hdf5-file with the name stored in the variable "filename"

user_input = input('Do you want to store the data in the hdf5 file "{}" (y/n)? '.format(filename))


if user_input == 'y':
    with h5py.File(input_path, 'w') as f:
    # Create a new group within the file
        group = f.create_group('Ge23')
        group2 = f.create_group('Ge23/waveform')
        group2.create_dataset('values',data=values_array) #stores pulse values in the waveform group
        group.create_dataset('timestamp',data=timestamps_array) #stores timestamp data in timestamp dataset
    print("Data saved!")
else:
    print("Data was not saved!")
    exit()

#--------------------------------------------------------------------------------------------------

#Runs the processing routine and produces a new hdf5-file with the name stored in the variable "output_filename"

user_input2 = input('Do you want to run the processing routine on hdf5 file "{}" (y/n)? '.format(filename))

if user_input2 == 'y':
    result = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    stdout, stderr = result.communicate()


   
    print(stdout)

        
    print(stderr)

else:
    print("Processing routine is not run!")
    exit()

#-------------------------------------------------------------------------------------------------

#prints the energy values determined by processing routine and plots the histogram 

user_input3 = input('Do you want to plot the energy for hdf5 file "{}" (y/n)? '.format(output_filename))

if user_input3 =='y':
    with h5py.File(output_path, "r") as f:
    # Print all root level object names (aka keys) 
    # these can be group or dataset names 
        print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[0]

    # get the object type for a_group_key: usually group or dataset
        print(type(f[a_group_key])) 

        data = list(f[a_group_key])
        print(data)

        ds_obj = f[a_group_key]      # returns as a h5py dataset object



        energy=np.asarray(f["Ge23/energy"][:-1])
        print(energy)

    plt.hist(energy, bins=50, edgecolor='black') 
    plt.xlabel('energy values')
    plt.ylabel('number of entries')
    plt.title(output_plot_filename)
    plt.savefig(output_plot_path)
    plt.show()
