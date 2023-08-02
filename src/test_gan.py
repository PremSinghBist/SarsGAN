"""Test GAN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow as tf
from common.bio.amino_acid import sequences_to_fasta
from common.bio.sequence import Sequence

from gan.models import get_model
from gan.parameters import get_flags
from gan.documentation import setup_logdir, get_properties
from gan.protein.helpers import convert_to_acid_ids
from common.model.ops import slerp
import numpy as np
import datetime
import os
from tensorflow.python.training.monitored_session import ChiefSessionCreator, MonitoredSession

slim = tf.contrib.slim
tfgan = tf.contrib.gan

flags.DEFINE_integer('n_seqs', 5, 'Number of sequences to be generated')
flags.DEFINE_boolean('use_cpu', True, 'Flags to determine whether to use CPU or not')
FLAGS = get_flags()

#FLAGS.DEFINE_integer('batch_size', 20, 'Number of sequences to be generated')







def main(_):
    '''
    After Training get completed, 
    Investigate GAN performance via interpolation. 
    -This will generate the sequences
    '''
    tf.logging.set_verbosity(tf.logging.INFO)
    if FLAGS.use_cpu:
        with tf.device('cpu:0'):
            interpolate()
    else:
        interpolate()


def interpolate():
    properties = get_properties(FLAGS)
    logdir = setup_logdir(FLAGS, properties)
     #how many seqs to generate is controlled by noise: (batch size: parameters.py)
    noise = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.z_dim])
    model = get_model(FLAGS, properties, logdir, noise)
    generated_seqs = get_generated_seqs(model)
    session_creator = ChiefSessionCreator(master='', checkpoint_filename_with_path=tf.train.latest_checkpoint(logdir))
    seqs = []
    with MonitoredSession(session_creator=session_creator, hooks=None) as session:
        #Random numbers uniformally distrubted between -1 and 1 with size = z_dim
        noise1 = np.random.uniform(-1, 1, FLAGS.z_dim) 
        noise2 = np.random.uniform(-1, 1, FLAGS.z_dim)
        #slerp: spherical linear interpolation 
        #np.linspace(starting value, stop values,, number of samples to generate) | ratio: Influencing factor
        n = np.stack([slerp(ratio, noise1, noise2) for ratio in np.linspace(0, 1, FLAGS.batch_size)])
        
        #generated_seqs :  generated sequences by operation --> get_generated_seqs(model) 
        #model.discriminator_fake : Output provided by discriminator model when generated_seqs were given as input
        #This output will tell, how "realistic" the generated data appears to the discriminator
        #{noise: n} : noise is palceholder tensor key and n is actual value
        #session.run : executes graph for "generated_seqs" tensor and "model.discriminator_fake" tensor
        #the returned values of run method are stored in  "results, d_scores"
        
        
        #feed_dict={noise: n} this means noise is a placeholder declared above or
        # Key that should take the value [n] while executing  session
        results, d_scores = session.run([generated_seqs, model.discriminator_fake], feed_dict={noise: n})
        for i in range(FLAGS.batch_size):
            seqs.append(Sequence(id=i, seq=results[i], d_score=d_scores[i]))
            
        print(sequences_to_fasta(seqs, properties['class_mapping'], escape=False, strip_zeros=True))
        fasta_seqs = sequences_to_fasta(seqs, properties['class_mapping'], escape=False, strip_zeros=True) 
        
        current_datetime = datetime.datetime.now()
        formated_date_path_name = "generated_"+current_datetime.strftime('%Y_%m_%d_%H_%M_%S')+".fasta"
        gen_file_path = os.path.join(FLAGS.generate_dir, formated_date_path_name)
        save_generated_fasta(fasta_seqs, gen_file_path)

def save_generated_fasta(fasta_seqs, file_path):
    print("File will be Generated to path: ", file_path)
    with open(file_path, "w+") as f:
        f.write(fasta_seqs)
    
    
def get_generated_seqs(model):
    if FLAGS.one_hot:
        # argMax: return the index of maximum value |  -1 means last axis
        generated_seqs = tf.squeeze(tf.argmax(model.fake_x, axis=-1)) 
    else:
        generated_seqs = convert_to_acid_ids(model.fake_x)
    return generated_seqs


if __name__ == '__main__':
    tf.app.run()
