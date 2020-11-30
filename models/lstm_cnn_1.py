class LSTMCNN(Model):
    def __init__(self, embedding_matrix, dropout, lstm_hidden, cnn_filter):
        super(LSTMCNN, self).__init__()
        
        self.embedding = Embedding(MAX_WORDS, EMBED_SIZE, weights=[embedding_matrix],
                             trainable=True)
        self.sp_dropout = SpatialDropout1D(dropout)
        self.lstm = Bidirectional(LSTM(lstm_hidden, return_sequences=True,
                                       return_state=False))
        self.conv_3 = Conv1D(cnn_filter, kernel_size=3, padding='valid',
                             kernel_initializer="glorot_uniform")
        self.avg_pool = GlobalAveragePooling1D()
        self.max_pool = GlobalMaxPooling1D()
        self.l1 = Dense(6, activation='sigmoid')
        
    def call(self, inp):
        x = self.embedding(inp)
        x = self.sp_dropout(x)
        x = self.lstm(x)
        x1 = self.conv_3(x)
        x = concatenate([self.avg_pool(x1), self.max_pool(x1)])
        x = self.l1(x)
        return x
        
