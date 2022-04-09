import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

import sys
sys.path.append("../../FedAvg_Experiment")
from utils.evaluate import *


class Seq2SeqAttention(nn.Module):
    def __init__(self, args, vocab_size, word_embeddings):
        super(Seq2SeqAttention, self).__init__()
        self.args = args

        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, self.args.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)

        # Encoder RNN
        self.lstm = nn.LSTM(input_size=self.args.embed_size,
                            hidden_size=self.args.hidden_size,
                            num_layers=self.args.hidden_layers,
                            bidirectional=self.args.bidirectional)

        # Dropout Layer
        self.dropout = nn.Dropout(self.args.dropout_keep)

        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.args.hidden_size * (1 + self.args.bidirectional) * 2,
            self.args.output_size
        )

        # Softmax non-linearity
        self.softmax = nn.Softmax()

    def apply_attention(self, rnn_output, final_hidden_state):
        '''
        Apply Attention on RNN output
        
        Input:
            rnn_output (batch_size, seq_len, num_directions * hidden_size): tensor representing hidden state for every word in the sentence
            final_hidden_state (batch_size, num_directions * hidden_size): final hidden state of the RNN
            
        Returns:
            attention_output(batch_size, num_directions * hidden_size): attention output vector for the batch
        '''
        hidden_state = final_hidden_state.unsqueeze(2)
        attention_scores = torch.bmm(rnn_output, hidden_state).squeeze(2)
        soft_attention_weights = F.softmax(attention_scores, 1).unsqueeze(2)  # shape = (batch_size, seq_len, 1)
        attention_output = torch.bmm(rnn_output.permute(0, 2, 1), soft_attention_weights).squeeze(2)
        return attention_output

    def forward(self, x):
        # print(x.shape)  # x.shape = (max_sen_len, batch_size)
        embedded_sent = self.embeddings(x)
        # print(embedded_sent.shape)  # embedded_sent.shape = (max_sen_len=146, batch_size=128, embed_size=300)
        ##################################### Encoder #######################################
        lstm_output, (h_n, c_n) = self.lstm(embedded_sent)
        # print(lstm_output.shape)  torch.Size([86, 128, 64])  # lstm_output.shape = (seq_len, batch_size, num_directions * hidden_size)
        # print(h_n.shape)  torch.Size([2, 128, 32])
        # print(c_n.shape)  torch.Size([2, 128, 32])

        # Final hidden state of last layer (num_directions, batch_size, hidden_size)
        batch_size = h_n.shape[1]
        h_n_final_layer = h_n.view(self.args.hidden_layers,
                                   self.args.bidirectional + 1,
                                   batch_size,
                                   self.args.hidden_size)[-1, :, :, :]
        # print('h_n_final_layer.shape: ', h_n_final_layer.shape)  h_n_final_layer.shape:  torch.Size([2, 128, 32])

        ##################################### Attention #####################################
        # Convert input to (batch_size, num_directions * hidden_size) for attention
        final_hidden_state = torch.cat([h_n_final_layer[i, :, :] for i in range(h_n_final_layer.shape[0])], dim=1)
        # print(final_hidden_state.shape)  torch.Size([128, 64])

        attention_out = self.apply_attention(lstm_output.permute(1, 0, 2), final_hidden_state)
        # Attention_out.shape = (batch_size, num_directions * hidden_size)

        # print("final_hidden_state.shape: ", final_hidden_state.shape)  final_hidden_state.shape:  torch.Size([128, 64])
        # print("attention_out.shape: ", attention_out.shape)  attention_out.shape:  torch.Size([128, 64])
        #################################### Linear #########################################
        concatenated_vector = torch.cat([final_hidden_state, attention_out], dim=1)
        # print(concatenated_vector.shape)  torch.Size([128, 128])
        final_feature_map = self.dropout(concatenated_vector)  # shape=(batch_size, num_directions * hidden_size + num_directions * hidden_size)
        # print(final_feature_map.shape)  torch.Size([128, 128])
        final_out = self.fc(final_feature_map)
        # print(final_out.shape)  torch.Size([128, 4])
        return self.softmax(final_out)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss(self, loss):
        self.loss = loss

    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2

    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []
        batch_loss = []
        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.args.max_epochs / 3)) or (epoch == int(2 * self.args.max_epochs / 3)):
            self.reduce_lr()
        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.text.cuda()
                y = (batch.label - 1).type(torch.cuda.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x)
            loss = self.loss(y_pred, y)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.data.cpu().numpy())
            batch_loss.append(loss.item())

            if i % 100 == 0:
                print(f"\t\t\tIter: {i + 1}")
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print(f"\t\t\t\tAverage training loss: {avg_train_loss:.5f}")
                losses = []

                # Evalute Accuracy on validation set
                val_accuracy = evaluate_model(self, val_iterator)
                print(f"\t\t\t\tVal Accuracy: {val_accuracy:.4f}")
                self.train()

        return train_losses, val_accuracies, batch_loss  # , batch_loss
