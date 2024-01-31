import os
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd

BLAST_DB_PATH = "/home/perm/ProteinGAN/data/cov_seqs/sars_db/full/cov_db_full"
ALIGN_SAVE_PATH = "/home/perm/ProteinGAN/data/cov_seqs/sars_db/raw_datasets"

def excecute_blast(query_path):
        '''
        Returns Blast_results, errror if any 
        Enzyme class #10: comma separated value format
        https://www.ncbi.nlm.nih.gov/books/NBK279684/
        qseqid : query sequence id 
        sseqid : subject sequence id (db id)   | gi : gene info identifier in genbank datbase 
        qstart : Start of alignment in query
        qend : End of alignment in query
        sstart : Start of alignment in subject
        send : End of alignment in subject
        qseq : Aligned part of query sequence
        sseq : Aligned part of subject sequence
        length : alignment length 
        nident : no of identical matches
        Ppos (similarity) : The "ppos" is a measure of the percentage of positions in the alignment where the residues are identical (positives). 
        10 qseqid score evalue pident '''
        
        blastp = subprocess.Popen(
                ['blastp', '-db', BLAST_DB_PATH, "-max_target_seqs", "1", "-outfmt", "10 qseqid sseqid qstart qend sstart send nident score evalue pident qseq sseq ppos",
                "-matrix", "BLOSUM45","-evalue", "1000",  "-query", query_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        results, err = blastp.communicate() #b'25,45,5.2,40.000\n60,44,6.0,23.077\ | it is a binary form data 
        text_results = results.decode('utf-8') #Decode binary into text format
        #print("Result:*******: ",text_results)
        #print("Error: ******", err)
        return text_results, err

def parse_alignment_result(text_results):
    '''
    Returns dataframe of aligned result
    '''
    qseqid, sseqid  = [], [] 
    qstart, qend = [], []
    sstart, send  = [], []
    nident = []
    score = []
    evalue = []
    pident = []
    qseq, sseq, ppos = [], [], []
    #25,28698,7,16,2,11,4,45,5.2,40.000,DDMTDQCNFW,PNITNLCPFW
    #qseqid sseqid qstart qend sstart send nident score evalue pident qseq sseq
    for row in text_results.split(os.linesep):
        records = row.split(",")
        # print("Record len: ", len(records))
        # print("Record:", records)
        if len(records) != 13:
            continue 
        qseqid.append(records[0])
        sseqid.append(records[1])
        qstart.append(records[2])
        qend.append(records[3])
        sstart.append(records[4])
        send.append(records[5])
        nident.append(records[6])
        score.append(records[7])
        evalue.append(records[8])
        pident.append(records[9])
        qseq.append(records[10])
        sseq.append(records[11])
        ppos.append(records[12])
        
    result_frame = {
        'qseqid' : qseqid,  'sseqid': sseqid,  'qstart' : qstart,
        'qend' : qend,  'sstart': sstart,  'send': send,
        'nident': nident,  'score': score,  'evalue': evalue,
        'pident': pident,  'qseq' : qseq,  'sseq' : sseq, 'ppos' : ppos
    }  
    return result_frame 

def save_alignment_result(result_frame, save_path):
    #qseqid sseqid qstart qend sstart send nident score evalue pident
    df = pd.DataFrame(result_frame)
    df.to_csv(save_path, index=False)
    print(f"Alignment results save successfully to csv: {save_path}")

def perform_alignment(query_path):
    
    '''
    This is a wrapper method that performs alignment of
    query sequences given in query path (Fasta file) againt BLastDB 
    and Saves the alignment output to CSV file. 
    '''
    #Align sequences and save results to csv |  |  query_path_fsta_gt_50
    result , error = excecute_blast(query_path)
    result_dataframe = parse_alignment_result(result)
    save_alignment_result(result_dataframe, 
                        ALIGN_SAVE_PATH+'/aligment_results_{}.csv'.format(os.path.basename(query_path).split(".")[0] ))


def get_fasta_records_from_csv(input_csv_path):
    '''
    This method reads fasta information from csv file
    Returns: window segment , start postion, end position, Full Sequence
    '''
    df = pd.read_csv(input_csv_path)
    # df.head()
    windows = df['window'].to_list()
    start_positions = df['start_pos'].to_list()
    end_positions = df['end_pos'].to_list()
    full_seqs = df['full_seq'].to_list()
    assert len(full_seqs), len(windows)
    return windows, start_positions, end_positions, full_seqs

def write_fasta_record(windows, start_positions, end_positions, full_seqs, fasta_save_path):
    records  = []
    for index in range(len(windows)):
        header =  windows[index] + ',' + str(start_positions[index]) +","  + str(end_positions[index]) 
        #SeqIO.SeqRecord(Seq(YourSEQ), ID, DESC) #All in string format 
        #, description = header
        record  = SeqIO.SeqRecord(Seq(full_seqs[index]), id = str(index))
        records.append(record)
    assert len(records), len(windows)

    #Write record 
    with open(fasta_save_path, "w") as output_handle:
        SeqIO.write(records, output_handle, "fasta")
    print("Fasta file saved Successfully to the path {}".format(fasta_save_path))
        
def convert_csv_to_fasta(query_path):
    print("Converting CSV to FASTA Query Path", query_path)
    #Read csv | sequence , score
    df = pd.read_csv(query_path)
    sequences  = df['sequence'].to_list()
    scores = df['score'].to_list()
    assert len(sequences), len(scores)
    
    fasta_record_dict = {}
    #Convert sequences to dictionary to remove duplicate sequences 
    #only unique sequence will be added 
    for seq, score in zip(sequences, scores):
        fasta_record_dict[seq] = score
        
    #Create Sequence records
    fasta_records = [] # you can use both {}
    index = 0
    for sequence, score in fasta_record_dict.items():
        record  = SeqIO.SeqRecord(Seq(sequence), str(index), description=str(score))
        index +=1 
        fasta_records.append(record)
        
    #Write to a fasta file 
    fasta_save_path = query_path.split(".")[0] +'.fasta'
    print(fasta_save_path)
    fasta_save_path
    with open(fasta_save_path, 'w') as handle:
        SeqIO.write(list(fasta_records), handle, "fasta") #convert set() to a list 
        
    print("Fasta file successfully saved to a path: ", fasta_save_path)

if __name__== "__main__":
    query_path_csv_gt_50 ="/home/perm/ProteinGAN/data/cov_seqs/sars_db/raw_datasets/fasta_12000_pred_score_filter_greater_than_50.csv"
    query_path_csv_le_50 = "/home/perm/ProteinGAN/data/cov_seqs/sars_db/raw_datasets/fasta_12000_pred_score_filter_less_than_equal_50.csv"
    
    query_path_fsta_gt_50 = "/home/perm/ProteinGAN/data/cov_seqs/sars_db/raw_datasets/fasta_12000_pred_score_filter_greater_than_50.fasta"
    query_path_fsta_le_50 = "/home/perm/ProteinGAN/data/cov_seqs/sars_db/raw_datasets/fasta_12000_pred_score_filter_less_than_equal_50.fasta"

    # convert_csv_to_fasta(query_path_csv_gt_50)
    # convert_csv_to_fasta(query_path_csv_le_50)
    
    perform_alignment(query_path_fsta_gt_50)
    perform_alignment(query_path_fsta_le_50)
    
    