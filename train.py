import torch

from text_loader import get_text_data_loader
from model import Transformer

if(__name__ == "__main__"):
    file_path = "shakespeare.rtf"
    batch_size = 4
    context_length = 100
    vocab_size = 33473
    num_epochs = 3
    data_loader = get_text_data_loader(batch_size=batch_size,
                                      context_length=context_length,
                                      file_path=file_path)
    model = Transformer(vocab_size=vocab_size,
                       num_blocks = 1,
                       emb_dim = 512,
                       dim_ff = 1024,
                       num_heads =4, 
                       context_length=context_length,
                       )
    optim = torch.optim.Adam(params = model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        running_loss = 0.
        print(f"Epoch {epoch+1}/{num_epochs}")
        for i, (input, target) in enumerate(data_loader):
            output = model(input)
            output = output.permute(0,2,1)
            optim.zero_grad()
            loss = loss_fn(output, target)
            loss.backward()
            running_loss += loss
        print("Epoch loss: ", running_loss/(i+1))
      

