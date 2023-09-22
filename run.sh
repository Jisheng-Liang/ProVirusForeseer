python predict.py -fasta_file_path $1.fasta -embedding_file_path $1.pt
python predict_score.py -pt_file_path $1.pt -out_file_path $1.npy
