#!/usr/bin/python3
#Collection of functions that are useful for plotting in python
#Mainly related to stuff that useful for formatting the plot, and does not need to be repated in every program
import matplotlib.pyplot as plt


def format_plot(fig,ax,xlabel,ylabel,**kwargs):
    #Pass fig and ax objects to format the plot
    
    #Set optional arguments
    title = kwargs.get('title',False)
    grid = kwargs.get('grid',True)
    fontsize = kwargs.get('fontsize',20)
    labelsize = kwargs.get('labelsize',20)

    #Format the plot
    ax.set_xlabel(xlabel,fontsize = fontsize)
    ax.set_ylabel(ylabel,fontsize = fontsize)
    if title:
        ax.set_title(title,fontsize = fontsize)
    ax.tick_params(axis = 'both', labelsize = labelsize)
    if grid:
        ax.grid()
    fig.tight_layout()
    return 
