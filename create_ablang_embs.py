from Bio import SeqIO
import torch
import os
import ablang2

def load_model_and_alphabet(device):
    model = ablang2.pretrained(model_to_use='ablang2-paired', random_init=False, ncpu=1, device=device)
    return model

def get_protein_embeddings(model, sequence, device):

    tokenized_seq = model.tokenizer([sequence], pad=True, w_extra_tkns=False, device=device)

    with torch.no_grad():
        results = model.AbRep(tokenized_seq).last_hidden_states
        return results[0].to('cpu')

def process_sequences(model, fasta_file, device, protein_dictionary):
    
    for no, record in enumerate(SeqIO.parse(fasta_file, "fasta"), 1):
        print(f"{no}", flush=True)

        id = record.id
        sequence = str(record.seq)

        protein_dictionary[id] = get_protein_embeddings(model, sequence, device)

def main():
    fasta_file = 'data/combined_sequences.fasta'
    output_dir = 'embeddings'
    protein_dictionary = {}

    device = torch.device('cuda')
    model = load_model_and_alphabet(device)

    process_sequences(model, fasta_file, device, protein_dictionary)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(protein_dictionary, os.path.join(output_dir, 'ablang.pt'))

if __name__ == "__main__":
    main()
