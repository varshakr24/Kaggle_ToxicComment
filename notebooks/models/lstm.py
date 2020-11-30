class LSTMCNN(Model):
    def __init__(self, embedding_matrix, dropout, lstm_hidden, cnn_filter):
        super(LSTMCNN, self).__init__()
        
        self.embedding = Embedding(MAX_WORDS, EMBED_SIZE, weights=[embedding_matrix],
                             trainable=True)
        self.sp_dropout = SpatialDropout1D(dropout)
        self.lstm = Bidirectional(LSTM(lstm_hidden, return_sequences=True,
                                       return_state=False))
        self.max_pool = GlobalMaxPooling1D()
        self.l2 = Dense(6, activation='sigmoid')
        
    def call(self, inp):
        x = self.embedding(inp)
        x = self.sp_dropout(x)
        x = self.lstm(x)
        x = self.max_pool(x)
        x = self.l2(x)
        return x
        
