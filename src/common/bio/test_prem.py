import subprocess
db_path = "/home/perm/ProteinGAN/data/cov_seqs/sars_db/cov_blast/covdb_val"
query_path="/home/perm/ProteinGAN/data/log/fasta.fasta"
# TODO: Enzyme class
blastp = subprocess.Popen(
    ['blastp', '-db', db_path, "-max_target_seqs", "1", "-outfmt", "10 qseqid score evalue pident",
        "-matrix", "BLOSUM45", "-query", query_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
results, err = blastp.communicate()
print("Reuslt: *****: ",results.decode())

print("Error: *****",err.decode())