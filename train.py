import torch
import tqdm
import numpy as np

from text_loader import get_text_data_loader
from text_encoding import decode_sequence, encode_sequence
from model import Transformer

if(__name__ == "__main__"):
    #check for gpu
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print (x)
    else:
        print ("MPS device not found.")
    file_path = "shakespeare.rtf"
    batch_size = 16
    context_length = 200
    vocab_size = 33473
    num_epochs = 2
    data_loader, word_to_index, index_to_word = get_text_data_loader(batch_size=batch_size,
                                      context_length=context_length,
                                      file_path=file_path)
    model = Transformer(vocab_size=vocab_size,
                       num_blocks = 1,
                       emb_dim = 128,
                       dim_ff = 512,
                       num_heads =2, 
                       context_length=context_length,
                       )
    optim = torch.optim.Adam(params = model.parameters(True), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    start_seq = encode_sequence(["KING", "HENRY"], word_to_index=word_to_index)
    start_seq = torch.LongTensor([start_seq]).to(torch.int64)
    print("Sampled before training: ", decode_sequence(model.sample_sequence(start_seq=start_seq, length_sequence=100).cpu().detach().numpy().tolist()[0], index_to_word))
    for epoch in range(num_epochs):
        running_loss = 0.
        print(f"Epoch {epoch+1}/{num_epochs}")
        for i, (input, target) in tqdm.tqdm(enumerate(data_loader)):
            output = model(input)
            output = output.permute(0,2,1)
            optim.zero_grad()
            loss = loss_fn(output, target)
            loss.backward()
            running_loss += loss
            optim.step()
        print("Epoch loss: ", running_loss/(i+1))
    print("Sampled after training: ", decode_sequence(model.sample_sequence(start_seq=start_seq, length_sequence=100).cpu().detach().numpy().tolist()[0], index_to_word))
