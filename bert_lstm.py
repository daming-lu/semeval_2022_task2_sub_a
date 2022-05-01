class bert_lstm(nn.Module):
    def __init__(self, bertpath, hidden_dim, output_size,n_layers,bidirectional=True, drop_prob=0.1):
        super(bert_lstm, self).__init__()
 
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        self.bert=BertModel.from_pretrained(bertpath)  # we use bert-base-multilingual-cased
        for param in self.bert.parameters():
            param.requires_grad = True
        
        # LSTM layers
        self.lstm = nn.LSTM(768, hidden_dim, n_layers, batch_first=True,bidirectional=bidirectional)
        
        # dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        # linear and sigmoid layers
        if bidirectional:
            self.fc = nn.Linear(hidden_dim*2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)
          
        #self.sig = nn.Sigmoid()
 
    def forward(self, x, hidden):
        batch_size = x.size(0)
        x=self.bert(x)[0]        
        _, (hidden_last, _) = self.lstm(x, hidden)        
        if self.bidirectional:
            hidden_last_L=hidden_last[-2]
            hidden_last_R=hidden_last[-1]
            hidden_last_out=torch.cat([hidden_last_L,hidden_last_R],dim=-1)
        else:
            hidden_last_out=hidden_last[-1]   #[32, 384]
            
        out = self.dropout(hidden_last_out)
        out = self.fc(out)
        return out
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        number = 1
        if self.bidirectional:
            number = 2
        
        if (USE_CUDA):
            hidden = (weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float().cuda(),
                      weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float().cuda()
                     )
        else:
            hidden = (weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float(),
                      weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().float()
                     )
        
        return hidden
