import torch
import tqdm

from text_loader import get_text_data_loader
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
    num_epochs = 3
    data_loader = get_text_data_loader(batch_size=batch_size,
                                      context_length=context_length,
                                      file_path=file_path)
    model = Transformer(vocab_size=vocab_size,
                       num_blocks = 1,
                       emb_dim = 256,
                       dim_ff = 512,
                       num_heads =2, 
                       context_length=context_length,
                       )
    optim = torch.optim.Adam(params = model.parameters(True), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
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
      

