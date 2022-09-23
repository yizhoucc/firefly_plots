# Plots for "Rationalizing Subject Behavior when Navigating in Virtual Reality" paper


# Structure:

## overall
the project is packed in docker and use vscode devcontainer's jupyter notebook for plot visualization. 
the vscode extentions are configured such that once the devcontainer is running, shift enter should send the code to generate plots in the interactive window.
the res_xx files are stand-alone. 
they are coresponding to the main result figures.

## firefly task
the gym file to define the firefly task.
this is a spatial navigation task in virtual reality space, where the subject is asked to navigate and stop at a transient visible 'firefly' target.
we think that the subject uses observation and internal prediction to achieve the path integration navigation. 


# Use:

## install docker and make the container

## run the plotting functions


# Future versions:
currently the docker image is a full image that orginally used for deploying on kubernetes clusters to run the computation.
we could remove some unused packages to compress the image.


# Other notes:
this work is done by yizhou @ xaq pitkow lab, Baylor College of Medicine / Rice university.




