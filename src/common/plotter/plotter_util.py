import os
import matplotlib.pyplot as plt
import numpy as np 
FIG_DIR = "/home/perm/ProteinGAN/src/common/plotter/fig"

def simple_plot(x_data, y_data, x_label, y_label, title, fig_name):
    plt.plot(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.savefig(FIG_DIR+os.path.sep+fig_name)
    plt.show()
    
    plt.clf()
    
def simple_plot_by_dictionary(dict_data, x_label, y_label, title, fig_name ):
    #As dictionary are unordered, we need to sort  it  by key
    plt.plot(*zip(*sorted(dict_data.items())))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.savefig(FIG_DIR+os.path.sep+fig_name)
    plt.show()
    
    plt.clf()
    
def plot_train_val_identity(steps, val_identity, train_identity, fig_name):
    plt.scatter(steps, val_identity, s = 5) #s: size
    plt.scatter(steps, train_identity, s = 5)
    
    #Create a fitting line for  validation and train identity data | perform linear regression
    val_slope, val_intercept = np.polyfit(steps, val_identity, 1)
    val_fitting_line = val_slope * np.array(steps) + val_intercept
    
    train_slope, train_intercept = np.polyfit(steps, train_identity, 1)
    train_fitting_line = train_slope * np.array(steps) + train_intercept
    
    plt.plot(steps, val_fitting_line, label='Validation Identity')
    plt.plot(steps, train_fitting_line, label='Train Identity')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Identity (%)')
    plt.legend() #Show legend
    plt.savefig(FIG_DIR+os.path.sep+fig_name)
    plt.show()
    
    plt.clf()
    
def plot_disc_gen_loss(steps, disc_loss, gen_loss, fig_name):

    last_train_step = max(steps)
    step_value = int(last_train_step/10) #Show just 10 steps  
    
    plt.plot(steps, disc_loss, label='Discriminator Loss')
    plt.plot(steps, gen_loss, label='Generator Loss')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend() #Show legend
    plt.savefig(FIG_DIR+os.path.sep+fig_name)
    plt.show()
    
    plt.clf()
    
    
    
def plot_sequence_identity_whisker(identities, steps, fig_name):
    '''
    This function performs box and whisker plotting for training and validating seq identity to 
    natural sequences. Train and validation identites are depicted using box plot
    ''' 
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.boxplot(identities, notch=True)

    ax.set_title('Identity Percentage of Generated Cov-Spike seqs to Natural Ones')
    ax.set_ylabel('Identity (%)')

    # Set the learning steps
    ax.set_xticklabels(steps, rotation=18)
    ax.set_xlabel('Learning Steps')

    save_comp_path = FIG_DIR+os.path.sep+fig_name
    plt.savefig(save_comp_path)
    plt.show()
    print("Figure saved successfull to path: ", save_comp_path)
    
    #Clear figure
    plt.clf()

def plot_sequence_identity_whisker1(identities, steps, fig_name):
    # Set the style
    plt.style.use('seaborn-whitegrid')

    # Define the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # create box with filled color
    bp = ax.boxplot(identities, notch=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('darkorange')
        
    #Change outliers colors to solid dots
    for outlier in bp['fliers']:
        outlier.set(marker='o', markersize=8, markerfacecolor='black',  linestyle='none')
        
        

    # Set labels and title
    ax.set_title('Identity of Generated Spikes to Natural Ones', fontsize=16)
    ax.set_ylabel('Identity (%)', fontsize=14)
    ax.set_xlabel('Learning Steps', fontsize=14)

    # Set x-axis labels with rotation
    ax.set_xticklabels(steps, rotation=45, ha='right', fontsize=12)

    # Adjust layout and padding
    plt.tight_layout()

    # Save and show the plot
    fig_name = 'boxplot_identity_1.png'
    save_comp_path = os.path.join(FIG_DIR, fig_name)
    plt.savefig(save_comp_path)
    plt.show()

    print("Figure saved successfully to path:", save_comp_path)
     
        

if __name__ == "__main__":
    
    print("Hellow world")
    