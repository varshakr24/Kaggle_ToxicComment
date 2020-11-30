from keras.models import Model
from keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D,GRU, Dropout, Embedding, Conv2D, GlobalMaxPooling1D, GlobalMaxPooling2D,  GlobalAveragePooling1D, concatenate, Dropout,SpatialDropout1D, Flatten, Reshape, Permute
from .constants import *
from transformers import TFBertModel

class BiLSTM(Model):
    def __init__(self, embedding_matrix, dropout, lstm_hidden, cnn_filter):
        super(BiLSTM, self).__init__()
        
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


class LSTM_CNN(Model):
    def __init__(self, embedding_matrix, dropout, lstm_hidden, cnn_filter):
        super(LSTM_CNN, self).__init__()
        
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
        x = self.sp_dropout(x)
        x1 = self.conv_3(x)
        x = concatenate([self.avg_pool(x1), self.max_pool(x1)])
        x = self.l1(x)
        return x


class GRU_CNN(Model):
    def __init__(self, embedding_matrix, dropout, lstm_hidden, cnn_filter):
        super(GRU_CNN, self).__init__()
        
        self.embedding = Embedding(MAX_WORDS, EMBED_SIZE, weights=[embedding_matrix],
                             trainable=True)
        self.sp_dropout = SpatialDropout1D(dropout)
        self.lstm = Bidirectional(GRU(lstm_hidden, return_sequences=True,
                                       return_state=False))
        self.conv_3 = Conv1D(cnn_filter, kernel_size=3, padding='valid',
                             kernel_initializer="glorot_uniform")
        self.avg_pool = GlobalAveragePooling1D()
        self.max_pool = GlobalMaxPooling1D()
        self.l2 = Dense(6, activation='sigmoid')
        
    def call(self, inp):
        x = self.embedding(inp)
        x = self.sp_dropout(x)
        x = self.lstm(x)
        x1 = self.conv_3(x)
        x = concatenate([self.avg_pool(x1), self.max_pool(x1)])

        x = self.l2(x)
        return x


class LSTM_CNN2(Model):
    def __init__(self, embedding_matrix, dropout, lstm_hidden, cnn_filter):
        super(LSTM_CNN2, self).__init__()
        
        self.embedding = Embedding(MAX_WORDS, EMBED_SIZE, weights=[embedding_matrix],
                             trainable=True)
        self.sp_dropout = SpatialDropout1D(dropout)
        self.lstm = Bidirectional(LSTM(lstm_hidden, return_sequences=True,
                                       return_state=False))
        self.conv_3 = Conv1D(cnn_filter, kernel_size=3, padding='valid',
                             kernel_initializer="glorot_uniform")
        self.conv_5 = Conv1D(cnn_filter, kernel_size=5, padding='valid',
                             kernel_initializer="glorot_uniform")
        self.avg_pool = GlobalAveragePooling1D()
        self.max_pool = GlobalMaxPooling1D()
        self.l1 = Dense(128, activation='sigmoid')
        self.l2 = Dense(6, activation='sigmoid')
        
    def call(self, inp):
        x = self.embedding(inp)
        x = self.sp_dropout(x)
        x = self.lstm(x)
        x1 = self.conv_3(x)
        x2 = self.conv_5(x)
        x = concatenate([self.avg_pool(x1), self.max_pool(x1), self.avg_pool(x2), self.max_pool(x2)])

        x = self.l1(x)
        x = self.l2(x)
        return x


class CNN_LSTM(Model):
    def __init__(self, embedding_matrix, dropout, lstm_hidden, cnn_filter):
        super(CNN_LSTM, self).__init__()
        
        self.embedding = Embedding(MAX_WORDS, EMBED_SIZE, weights=[embedding_matrix],
                             trainable=True)
        self.sp_dropout = SpatialDropout1D(dropout)
        self.lstm = Bidirectional(LSTM(lstm_hidden, return_sequences=True,
                                       return_state=False))
        self.conv_3 = Conv2D(cnn_filter, kernel_size=3, padding='valid',
                             kernel_initializer="glorot_uniform")
        self.max_pool = GlobalMaxPooling1D()
        self.l1 = Dense(6, activation='sigmoid')
        
    def call(self, inp):
        
        x = self.embedding(inp)
        x = self.sp_dropout(x)
        x = Reshape((MAX_LEN, EMBED_SIZE, 1))(x)
        x = self.conv_3(x)
        _, w, h, d = x.shape
        x = Permute((1,3,2))(x)
        x = Reshape((w, h * d))(x)
        x = self.lstm(x)
        x = self.sp_dropout(x)
        x = self.max_pool(x)
        x = self.l1(x)
        return x


class CNN_Large(Model):
    def __init__(self, embedding_matrix, dropout, lstm_hidden, cnn_filter):
        super(CNN_Large, self).__init__()
        
        self.embedding = Embedding(MAX_WORDS, EMBED_SIZE, weights=[embedding_matrix],
                             trainable=True)
        self.sp_dropout = SpatialDropout1D(dropout)
        self.conv_3 = Conv2D(cnn_filter, kernel_size=3, padding='valid',
                             kernel_initializer="glorot_uniform")
        self.conv_5 = Conv2D(cnn_filter, kernel_size=5, padding='valid',
                             kernel_initializer="glorot_uniform")
        self.conv_7 = Conv2D(cnn_filter, kernel_size=7, padding='valid',
                             kernel_initializer="glorot_uniform")
        self.avg_pool = GlobalAveragePooling1D()
        self.max_pool = GlobalMaxPooling2D()
        self.l1 = Dense(6, activation='sigmoid')
        self.dropout = Dropout(dropout)
        
    def call(self, inp):
        
        x = self.embedding(inp)
        x = self.sp_dropout(x)
        x = Reshape((MAX_LEN, EMBED_SIZE, 1))(x)
        x1 = self.conv_3(x)
        x2 = self.conv_5(x)
        x3 = self.conv_7(x)
        x = concatenate([self.max_pool(x1), self.max_pool(x2), self.max_pool(x3)])
        x = Flatten()(x)
        x = self.dropout(x)
        x = self.l1(x)
        return x
        

class BertModel(Model):    
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.l1 = Dense(128, activation='relu')
        self.l2 = Dense(6, activation='sigmoid')
        
    def call(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        x = self.bert(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        x = x[1]
        x = self.l2(self.l1(x))
                
        return  x
