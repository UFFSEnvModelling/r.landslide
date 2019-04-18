#!/usr/bin/env python
#############################################################
#
#  MODULE: r.landslide
#
#  AUTHOR(S): Lucimara Bragagnolo -------- lucimarabragagnolo@hotmail.com
#             Roberto Valmir da Silva --------- roberto.silva@uffs.edu.br
#             Jose Mario Vicensi Grzybowski - jose.grzybowski@uffs.edu.br
#
#  PURPOSE: Uses r.landslide for identification of areas 
#           susceptible to landslides
#
#  COPYRIGHT: (C) 2019 by the GRASS Development Team
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#############################################################


#%module: r.landslide
#% description: Creates a landslide susceptibility map
#% keyword: raster
#%end

#%option
#% key: num_hidden
#% type: integer
#% label: Number of hidden neurons
#% answer: 12
#% guisection: ANN parameters
#% required: no
#%end

#%option
#% key: lr_rate
#% type: double
#% label: Learning rate (0-1)
#% answer: 0.6
#% guisection: ANN parameters
#% required: no
#%end

#%option
#% key: nr_epochs
#% type: integer
#% label: Number of epochs
#% answer: 200
#% guisection: ANN parameters
#% required: no
#%end

#%option
#% key: val_samples
#% type: double
#% label: Percentage of data for validation (0-1)
#% answer: 0.15
#% guisection: ANN parameters
#% required: no
#%end

#%option
#% key: test_samples
#% type: double
#% label: Percentage of data for test (0-1)
#% answer: 0.15
#% guisection: ANN parameters
#% required: no
#%end

#%flag
#% key: b
#% label: Train a set of ANNs and select the best one [batch mode]
#% guisection: ANN parameters
#%end

#%option
#% key: min_hidden
#% type: integer
#% label: [batch mode] Minimum number of hidden neurons
#% answer: 2
#% guisection: ANN parameters
#% required: no
#%end

#%option
#% key: max_hidden
#% type: integer
#% label: [batch mode] Maximum number of hidden neurons
#% guisection: ANN parameters
#% answer: 0
#% required: no
#%end

#%option
#% key: init_cond
#% type: integer
#% label: [batch mode] Number of initial conditions
#% description: Number of times the synaptic weights are restarted
#% guisection: ANN parameters
#% answer: 1
#% required: no
#%end

#%flag
#% key: s
#% label: Perform ONLY the training, validation and test steps
#% guisection: Training
#%end

#%option G_OPT_R_INPUT
#% key: layers
#% multiple: yes
#% label: Thematic layers
#% description: Insert rasters of environmental parameters that influence the occurrence of landslides
#% guisection: Training
#% required: no
#%end

#%option G_OPT_V_INPUT
#% key: landslide
#% label: Point vector with landslides locations
#% description: Vector layer containing the location of landslides points
#% required: no
#% guisection: Training
#%end

#%option G_OPT_V_INPUT
#% key: nolandslide
#% label: Point vector with non-landslides locations
#% description: Vector layer containing the location of non-landslides points
#% required: no
#% guisection: Training
#%end

#%option G_OPT_F_INPUT
#% key: coord
#% label: [optional] Text file with X,Y coordinates of landslides locations
#% description: Text file containing the landslides points coordinates
#% required: no
#% guisection: Training
#% required: no 
#%end

#%option G_OPT_F_SEP
#% key: separator
#% label: Character separator of text file containing the X,Y coordinates
#% description: Character separator of text file containing the X,Y coordinates
#% required: no
#% guisection: Training
#% answer: comma
#% required: no 
#%end

#%option G_OPT_R_OUTPUT
#% key: output
#% description: Name of output susceptibility map
#% guisection: Training
#% required: no
#%end

#%option G_OPT_F_OUTPUT
#% key: directory
#% label: Directory name to save files (it will be created)
#% description: Directory name to save the ANN training results
#% required: no
#% guisection: Training
#%end

#%flag
#% key: r
#% description: Perform ONLY the application
#% guisection: Application
#%end

#%option G_OPT_M_DIR
#% key: direc_reckon
#% description: Input path directory containing weights, peights and bias text files
#% guisection: Application
#% required: no
#%end

#%option G_OPT_R_OUTPUT
#% key: outputr
#% description: Name of output susceptibility map
#% guisection: Application
#% required: no
#%end

#%flag
#% key: f
#% description: Save susceptibility raster in directory
#%end

#%option G_OPT_R_INPUT
#% key: dregion
#% label: Insert a rast map to define a temporary region to reckoning step
#% required: no
#%end

import sys
import grass.script as gscript
from grass.script import array as garray
import numpy as np
from ANN_module import ann_module 
from ANN_module import ann_reckon
from ANN_module import ANN_batch
import os

def main():
    options, flags = gscript.parser()

    #Output
    raster_final = options['output']

    #Inputs
    nr_epochs = int(options['nr_epochs'])
    num_hidden = int(options['num_hidden'])
    coef = float(options['lr_rate'])
    val_samples = float(options['val_samples'])
    test_samples = float(options['test_samples'])
    min_hidden = int(options['min_hidden'])
    max_hidden = int(options['max_hidden'])
    trials = int(options['init_cond'])
    directory = options['directory']
    flag_batch = flags['b'] #To train a set of ANNs
    flag_train = flags['s']
    flag_reckon = flags['r']
    flag_save = flags['f']
    direc_reckon = options['direc_reckon']
    coord = options['coord']
    tregion = options['dregion']
   
    
    #Parameters verification
    if num_hidden == 0:
        gscript.fatal(_("Zero is not a valid value for number of hidden neuros."))
   
    if nr_epochs == 0:
        gscript.fatal(_("Zero is not a valid value for number of epochs."))

    if coef == 0:
        gscript.fatal(_("Zero is not a valid value for learning rate."))
    
    if coef >= 1:
        gscript.fatal(_("This is not a valid value for the learning rate. Please enter a value within the range 0-1."))

    if val_samples == 0:
        gscript.fatal(_("Zero is not a valid value to create the validation dataset."))

    if val_samples >= 1:
        gscript.fatal(_("This is not a valid value to create the validation dataset."))

    if trials == 0:
        gscript.fatal(_("This is not a valid value for the number of initial conditions."))

    if flag_train and flag_reckon:
        gscript.fatal(_("The both flags Perform ONLY reckoning and Perform ONLY training are selected. Please uncheck one."))

    if flag_batch and flag_reckon:
        gscript.fatal(_("The both flags Train a set of ANNs and Perform ONLY reckoning are selected. Please uncheck one."))

    if test_samples == 0:
        gscript.fatal(_("Zero is not a valid value to create the test dataset."))

    if val_samples >= 1:
        gscript.fatal(_("This is not a valid value to create the test dataset."))

    if flag_batch:
        if min_hidden == 0:
            min_hidden = 2
            gscript.warning(_("The minimum number of hidden neurons was defined as 2"))

        if max_hidden == 0:
            gscript.warning(_("The maximum number of hidden neurons will be calculated based on the number of input parameters."))

    if val_samples > 0.20:
        gscript.warning(_("A high percentage for validation data reduces the number of training records. Consider reducing the value of this parameter."))

    if test_samples > 0.20:
        gscript.warning(_("A high percentage for test data reduces the number of training records. Consider reducing the value of this parameter."))


    #print flag_reckon
    work_dir = os.getcwd() #Get the current working directory

    if flag_reckon is False:
        #Get the input map layers
        rasters = options['layers'].split(",") #get the raster maps names (comma separated)
        #split: it separates the maps names and creates a list

        n = len(rasters) #number of input maps
        array_maps = []

        #Creates a temporary region for calculation
        if tregion:
            gscript.use_temp_region()
            gscript.run_command('g.region', raster=tregion, align=rasters[0])

            #Create a matrix with the map layers for reckon with tregion
            for i in range(0,n):
                a = garray.array(mapname=rasters[i],null=-9999) #this function does not aceppt NAN values
                a[a==-9999]=np.nan #replacing to NaN
                array_maps.append(a) #List with all maps 

            row = int(a.shape[0])
            col = int(a.shape[1])
            lines = int(row*col)

            reckon = np.zeros((lines,n))
            for i in range(0,n):
                reckon[:,i] = np.asarray(array_maps[i].reshape((lines,)))
            
            gscript.del_temp_region()

        else:
            #Create a matrix with the map layers for reckon without region
            for i in range(0,n):
                a = garray.array(mapname=rasters[i],null=-9999) #this function does not aceppt NAN values
                a[a==-9999]=np.nan #replacing to NaN
                array_maps.append(a) #List with all maps 

            row = int(a.shape[0])
            col = int(a.shape[1])
            lines = int(row*col)

            reckon = np.zeros((lines,n))
            for i in range(0,n):
                reckon[:,i] = np.asarray(array_maps[i].reshape((lines,)))

        if not coord:
            landslide = options['landslide']
        else:
            separator = options['separator']
            gscript.run_command(
                "v.in.ascii",
                input=coord,
                output='landslide_points',
                separator=separator,
                overwrite=True)
            landslide = 'landslide_points'

        #Get the vector points layers
        nolandslide = options['nolandslide']

        #Collect the information from the rasters (v.what.rast)
        vecname = landslide,nolandslide

        columnst = []
        for i in range(n):
            r = rasters[i] #Get the raster name
            c = r #Vector column name is the same that raster name
            if c.endswith('@PERMANENT'):
                c = c[:-10]
            columnst.append(c) #Vector columns title
            for j in range(0,2): #vectors landslide/no-landslide
                v = vecname[j] #Landslides or nonlandslide
                gscript.run_command(
                    "v.what.rast", 
                    map=v,
                    raster=r,
                    column=c
                    )   

        tempfile = gscript.tempfile() #Creates a temporary file
        gscript.run_command(
            "v.out.ascii",
            input=landslide,
            output=tempfile+'.txt',
            columns=columnst,
            format='point',
            separator='comma'
            )

        land = np.genfromtxt(tempfile+'.txt', delimiter=",") #This command permits open csv with missing values
        land = np.delete(land,[0,1,2],axis=1) #Delete columns with coords and ID

        nanvalues = np.argwhere(np.isnan(land)) #Checking for nan values
        
        if nanvalues.size > 0:
            k = 0
            nanvalues = np.argwhere(np.isnan(land))
            while nanvalues.size > 0:
                k = k + 1
                land = np.delete(land, (int(nanvalues[0,0])), axis=0) #Delete the line with nan
                nanvalues = np.argwhere(np.isnan(land))

            gscript.warning("Some rows with null values of landslide points were deleted.")

        land = np.append(land,np.ones((land.shape[0],1)),axis=1) #Add column of ones (landslides)

        tempfile2 = gscript.tempfile()
        gscript.run_command(
           "v.out.ascii",
           input=nolandslide,
           output=tempfile2+'.txt',
           columns=columnst,
           format='point',
           separator='comma'
            )

        noland = np.genfromtxt(tempfile2+'.txt', delimiter=",")
        noland = np.delete(noland,[0,1,2],axis=1)

        nanvalues = np.argwhere(np.isnan(noland)) #Checking for nan values

        if nanvalues.size > 0:
            k = 0
            nanvalues = np.argwhere(np.isnan(noland))
            while nanvalues.size > 0:
                k = k + 1
                noland = np.delete(noland, (int(nanvalues[0,0])), axis=0) #Delete the line with nan
                nanvalues = np.argwhere(np.isnan(noland))

            gscript.warning("Some rows with null values of nonlandslide points were deleted.")
        
        noland = np.append(noland,np.zeros((noland.shape[0],1)),axis=1) #Add column of zeros (non landslides)

        #Join tables
        dataset = np.append(land,noland,axis=0)
        np.random.shuffle(dataset) #Randomize dataset
        
        #Input and outuput data from training, validation and test
        input_data = dataset[:,0:n]
        output_data = dataset[:,n]

        if not flag_batch: #Just one neural network trained
            final = ann_module(input_data,output_data,reckon,num_hidden,coef,nr_epochs,
                val_samples,test_samples,directory,columnst,col,row,flag_train)

        else: #Flag_batch is true
            if max_hidden == 0:
                max_hidden = n*2+1

            hidden = np.arange(min_hidden,max_hidden+1,1)
            
            gscript.message(_("Definitions for batch mode: "))
            gscript.message(_("Hidden neurons: "))
            gscript.message(hidden)
            gscript.message(_("Number of initial conditions: "))
            gscript.message(trials)

            total_ANNs = len(hidden)*trials
            gscript.message(_("Number of ANNs that will be tested: "))
            gscript.message(total_ANNs)
            gscript.warning(_("This may take a while!"))
    
            final = ANN_batch(input_data,output_data,reckon,hidden,trials,coef,
                nr_epochs,val_samples,test_samples,directory,columnst,col,row,flag_train)

        #Save rasters name in order
        os.chdir(directory)
        with open("rasters_names.txt", "w") as my_file:
            my_file.write(options['layers'])
        os.chdir(work_dir)

        if not flag_train: #Generates final map (is not just training)

            #In case there is temp region
            if tregion:
                gscript.use_temp_region()
                gscript.run_command('g.region', raster=tregion, align=rasters[0])
                
                suscep = garray.array()
                for i in range(0,final.shape[0]):
                    for j in range(0,final.shape[1]):
                        suscep[i,j] = final[i,j]

                suscep.write(mapname=raster_final, overwrite=True) #Create raster file

                if flag_save: #Save final raster in the directory
                    os.chdir(directory)
                    gscript.run_command(
                        "r.out.gdal",
                        input=raster_final,
                        output=str(raster_final)+'.tiff',
                        format='GTiff',
                        type='Float64'
                        )
                    os.chdir(work_dir)
                gscript.del_temp_region()

            else:
                suscep = garray.array()
                for i in range(0,final.shape[0]):
                    for j in range(0,final.shape[1]):
                        suscep[i,j] = final[i,j]

                suscep.write(mapname=raster_final, overwrite=True) #Create raster file

                if flag_save: #Save final raster in the directory
                    os.chdir(directory)
                    gscript.run_command(
                        "r.out.gdal",
                        input=raster_final,
                        output=str(raster_final)+'.tiff',
                        format='GTiff',
                        type='Float64'
                        )
                    os.chdir(work_dir)

    else: #flag_reckon is True

        raster_final = options['outputr']

        os.chdir(direc_reckon)
        weights = np.genfromtxt('weights.txt', delimiter=",") 
        peights = np.genfromtxt('peights.txt', delimiter=",") 
        peights = np.reshape(peights,(peights.shape[0],1))
        biasH = np.genfromtxt('biasH.txt', delimiter=",") 
        biasH = np.reshape(biasH,(biasH.shape[0],1))
        biasO = np.genfromtxt('biasO.txt', delimiter=",") 
        biasO = np.reshape(biasO,(1,1))
        max_in = np.genfromtxt('max_in.txt', delimiter=",") 
        max_in = np.reshape(max_in,(1,max_in.shape[0]))
        min_in = np.genfromtxt('min_in.txt', delimiter=",") 
        min_in = np.reshape(min_in,(1,min_in.shape[0]))

        with open('rasters_names.txt', 'r') as myfile:
          rasters = myfile.read()
        os.chdir(work_dir)

        #Get the input map layers
        rasters = rasters.split(",") #get the raster maps names (comma separated)
        #split: it separates the maps names and creates a list
        n = len(rasters) #number of input maps
        array_maps = []

        #Creates a temporary region for calculation
        if tregion:
            gscript.use_temp_region()
            gscript.run_command('g.region', raster=tregion, align=rasters[0])

            #Create a matrix with the map layers for reckon with tregion
            for i in range(0,n):
                a = garray.array(mapname=rasters[i],null=-9999) #this function does not aceppt NAN values
                a[a==-9999]=np.nan #replacing to NaN
                array_maps.append(a) #List with all maps 

            row = int(a.shape[0])
            col = int(a.shape[1])
            lines = int(row*col)

            reckon = np.zeros((lines,n))
            for i in range(0,n):
                reckon[:,i] = np.asarray(array_maps[i].reshape((lines,)))
            
            gscript.del_temp_region()

        else:
            #Create a matrix with the map layers for reckon without region
            for i in range(0,n):
                a = garray.array(mapname=rasters[i],null=-9999) #this function does not aceppt NAN values
                a[a==-9999]=np.nan #replacing to NaN
                array_maps.append(a) #List with all maps 

            row = int(a.shape[0])
            col = int(a.shape[1])
            lines = int(row*col)

            reckon = np.zeros((lines,n))
            for i in range(0,n):
                reckon[:,i] = np.asarray(array_maps[i].reshape((lines,)))

        for j in range(0,reckon.shape[1]): 
            if max_in[0,j] != 0:
                reckon[:,j] = (reckon[:,j] - min_in[0,j])/(max_in[0,j]-min_in[0,j])
        
        output_reckon = ann_reckon(reckon,weights,peights,biasH,biasO)

        final = np.reshape(output_reckon,(col,row),order='F')
        final = np.transpose(final)

        #In case there is temp region
        if tregion:
            gscript.use_temp_region()
            gscript.run_command('g.region', raster=tregion, align=rasters[0])
            
            suscep = garray.array()
            for i in range(0,final.shape[0]):
                for j in range(0,final.shape[1]):
                    suscep[i,j] = final[i,j]

            suscep.write(mapname=raster_final, overwrite=True) #Create raster file

            if flag_save: #Save final raster in the directory
                os.chdir(directory)
                gscript.run_command(
                    "r.out.gdal",
                    input=raster_final,
                    output=str(raster_final)+'.tiff',
                    format='GTiff',
                    type='Float64'
                    )
                os.chdir(work_dir)
            gscript.del_temp_region()

        else:
            suscep = garray.array()
            for i in range(0,final.shape[0]):
                for j in range(0,final.shape[1]):
                    suscep[i,j] = final[i,j]

            suscep.write(mapname=raster_final, overwrite=True) #Create raster file

            if flag_save: #Save final raster in the directory
                os.chdir(directory)
                gscript.run_command(
                    "r.out.gdal",
                    input=raster_final,
                    output=str(raster_final)+'.tiff',
                    format='GTiff',
                    type='Float64'
                    )
                os.chdir(work_dir)

    return 0
#Do not use the print statement (print function in Python 3) for informational output. This is reserved for standard module output if it has one. 
#https://trac.osgeo.org/grass/wiki/Submitting/Python

if __name__ == "__main__":
    sys.exit(main())
