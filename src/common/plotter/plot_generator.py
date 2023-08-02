import os 
# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Disable XLA compilation
os.environ["TF_XLA_FLAGS"] = ""

import tensorflow as tf
from tensorflow.summary import FileWriter
import datetime
import plotter_util as plt_util

tag_dict = {}

def read_summary_events(log_dir):
    sess = tf.Session()
    summary_writer = FileWriter(log_dir, sess.graph)
    events = tf.train.summary_iterator(log_dir)
    return events
def read_multiple_log_events(log_dir):
    '''
    The first event file contains the most important logs
    '''
    event_files = tf.train.match_filenames_once(log_dir + '/events.out.tfevents.*')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(event_files.initializer)
        file_paths = sess.run(event_files)
        for file_path in file_paths:
            print("Reading events from:", file_path)
            for event in tf.train.summary_iterator(file_path):
                # Process the event data
                print(event)


    
def build_tag_dictionary(events):
    initialize_tag_dict()
    '''
    Returns steps and validatation 
    '''
    for event in events:
        for value in event.summary.value:
            add_tag(event.step, value.tag, value.simple_value )
           
def initialize_tag_dict():
    '''
    Intialize the available tags 
    '''
    tag_dict['BLAST_train'] = {}
    tag_dict['Blast/val/Identity'] = {}
    tag_dict['Blast/train/Identity'] = {} 
    tag_dict['Blast/val/Evalue'] = {}
    tag_dict['Blast/train/Evalue'] = {}
    tag_dict['Blast/train/BLOMSUM45'] = {}
    tag_dict['Blast/val/BLOMSUM45'] = {}
    tag_dict['Stddev/model/stddev/Stddev/real'] = {}
    tag_dict['Stddev/model/stddev/Stddev/fake'] = {}
    tag_dict['1_loss/model/tensorboard/1_loss/d_loss'] = {}
    tag_dict['1_loss/model/tensorboard/1_loss/g_loss'] = {}
    tag_dict['2_loss_component/model/tensorboard/2_loss_component/d_loss_real'] = {}
    tag_dict['2_loss_component/model/tensorboard/2_loss_component/d_loss_fake'] = {}
    tag_dict['3_discriminator_values/model/tensorboard/3_discriminator_values/d_real'] = {}
    tag_dict['3_discriminator_values/model/tensorboard/3_discriminator_values/d_fake'] = {}
    tag_dict['global_step/sec'] = {}
    
def get_tag_dictionary():
    return tag_dict   

def add_tag(step, tag_name, tag_value):
    if tag_name in tag_dict:
        value_dict =  tag_dict[tag_name]   
        #Add key and value in value dictionary
        value_dict[step] =   tag_value
        #assign the updated value 
        tag_dict[tag_name]  = value_dict
        
def build(events):
    '''
    Wrapper to build tag dictionary from events
    '''
    build_tag_dictionary(events)
    tag_dict = get_tag_dictionary()
    return tag_dict
    
def plot_sequence_identity_whisker(tag_dict, fig_name):
    '''
    It Plots whisker and box plot using identity metrics for validation-and-training-generated-sequences  
    compared to natural sequences
    -We obtain identity record at the interval of 1200. 
    -Box plots are created at  each interval which is  doubled at every step : eg: 1200, 2400, 4800, 9600 
    -All  identities values found in between step i and i+1  will be used to 
    create a single box plot 
    
    
    
    '''
    val_identity_dict = tag_dict['Blast/val/Identity']
    train_identity_dic = tag_dict['Blast/train/Identity']
    steps = []
    identities = []
    global_step_track_no  = 1200
    train_val_identity = []
    
    for step in val_identity_dict:
        if step < global_step_track_no :
            train_val_identity.append(val_identity_dict[step])
            if step in train_identity_dic:
                train_val_identity.append(train_identity_dic[step])
        else:
            if len(train_val_identity) > 0:
                identities.append(train_val_identity)
                steps.append(step)
                train_val_identity = []
            global_step_track_no  = global_step_track_no * 2
            
    plt_util.plot_sequence_identity_whisker(identities, steps, fig_name) 

def plot_train_val_identity(tag_dict, fig_name):
    val_identity_dict = tag_dict['Blast/val/Identity']
    train_identity_dic = tag_dict['Blast/train/Identity']
    
    '''
    To maintain the consistency in train and validation records corresponding to the same step,
    we iterate over train dictionary and check corresponding key in validation dictionary
    -Do not add records for plot when the data is not available for both 
    '''
    val_identity = []
    train_identity = []
    steps = []
    for step in train_identity_dic.keys():
        if step in val_identity_dict:
            val_identity.append(val_identity_dict[step])
            train_identity.append(train_identity_dic[step])
            steps.append(step)
        
    #val_identity = [identity_val for  identity_val in val_identity_dict.values()] 
    #train_identity = [identity_val for  identity_val in train_identity_dic.values()] 
    #steps = [step for  step in train_identity_dic.keys()] 
    
    plt_util.plot_train_val_identity(steps, val_identity, train_identity, fig_name)
    
                
def plot_disc_gen_loss(tag_dict, fig_name):
    desc_loss_dict = tag_dict['1_loss/model/tensorboard/1_loss/d_loss'] 
    gen_loss_dict = tag_dict['1_loss/model/tensorboard/1_loss/g_loss'] 
    
     #Create arrays of desc loss and generator loss 
    desc_loss = [loss for step, loss in desc_loss_dict.items()] 
    steps = [step  for step, loss in desc_loss_dict.items()] 
    gen_loss =  [loss for step, loss in gen_loss_dict.items()]
    
    plt_util.plot_disc_gen_loss(steps, desc_loss, gen_loss, fig_name)
            
def save_tag_dict_to_json(tag_dict):
    import json
    with open("tag_dict.json", 'w') as f:
        json.dump(tag_dict, f)
        
if __name__ == "__main__":
    #print("TensorFlow version:", tf.__version__)
    log_dir = "/home/perm/ProteinGAN/data/log/events.out.tfevents.1690681956.nscl-2"
    events = read_summary_events(log_dir)
    tag_dict = build(events)
    #save_tag_dict_to_json(tag_dict)
    
    current_datetime = datetime.datetime.now()
    label = current_datetime.strftime("%d_%b_%H_%M_%S_%f_non_sat")[:-3] 

    plot_sequence_identity_whisker(tag_dict, f'seq_identity_whisker_{label}.png')
    plot_disc_gen_loss(tag_dict, f'disc_gen_loss_{label}.png')
    plot_train_val_identity(tag_dict, f'train_val_identity_{label}.png')
    
       
    
    
    
 
    
